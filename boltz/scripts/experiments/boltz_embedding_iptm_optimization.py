#!/usr/bin/env python3
"""Boltz-2 embedding-space optimization to maximize 0.8*iPTM + 0.2*pTM.

This is analogous to `boltz_embedding_nmr_optimization.py`, but the objective
is a confidence metric computed by Boltz itself:

  objective = 0.8 * iptm + 0.2 * ptm

We optimize embeddings (s_opt, z_opt) directly, by unrolling the diffusion
sampling loop step-by-step and backpropagating through:
  embeddings -> diffusion_conditioning -> denoiser (x0_hat) -> confidence_module -> (iptm, ptm)

Note:
- This requires that the loaded Boltz2 checkpoint has `confidence_module` and `distogram_module`.
- Optimization is skipped for diffusion steps >= stop_opt_after_t (default 160) to reduce
  distribution shift late in denoising.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True  # optional but consistent
torch.set_float32_matmul_precision("high")  # optional (PyTorch 2.x)


from torch import Tensor
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Coords, Interface, Manifest, StructureV2
from boltz.data.write.mmcif import to_mmcif
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzProcessedInput,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
    download_boltz2,
    process_inputs,
)
from boltz.model.loss.diffusionv2 import weighted_rigid_align
from boltz.model.models.boltz2 import Boltz2

LOG = logging.getLogger("boltz_embedding_iptm_optimization")


@dataclass
class OptimConfig:
    sequences: list[str]
    pdb_id: str
    chain_ids: list[str]
    out_dir: Path = Path("./boltz_embedding_iptm_optim")
    cache: Path = Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()

    # Diffusion
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1  # unused in step-by-step loop; kept for interface parity

    # Optimization
    outer_steps: int = 1
    inner_steps: int = 1
    stop_opt_after_t: int = 160
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    batch_size: int = 5
    optimize_s: bool = False  # default z-only; s grads may be missing with kernels

    # Objective weights
    w_iptm: float = 0.8
    w_ptm: float = 0.2

    # W&B
    use_wandb: bool = True
    wandb_project: str = "boltz-embedding-iptm-optim"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

    seed: int = 42


def create_yaml(
    sequences: list[str],
    chain_ids: list[str],
    path: Path,
    template_cif: Path | None,
    msa: str | None,
) -> None:
    data: dict[str, Any] = {
        "version": 1,
        "sequences": [
            {"protein": {"id": cid, "sequence": seq}}
            for cid, seq in zip(chain_ids, sequences, strict=True)
        ],
    }
    if msa is not None:
        # If provided, apply the same MSA path to each chain entry (rarely used here).
        for entry in data["sequences"]:
            entry["protein"]["msa"] = msa
    if template_cif is not None:
        data["templates"] = [{"cif": str(template_cif.resolve())}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def process_single_input(
    input_path: Path,
    out_dir: Path,
    cache: Path,
    use_msa_server: bool,
    msa_server_url: str,
) -> BoltzProcessedInput:
    out_dir.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)
    process_inputs(
        data=[input_path],
        out_dir=out_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=cache / "mols",
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy="greedy",
        boltz2=True,
        preprocessing_threads=1,
        max_msa_seqs=8192,
    )
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    processed_dir = out_dir / "processed"
    return BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(processed_dir / "constraints" if (processed_dir / "constraints").exists() else None),
        template_dir=(processed_dir / "templates" if (processed_dir / "templates").exists() else None),
        extra_mols_dir=(processed_dir / "mols" if (processed_dir / "mols").exists() else None),
    )


def load_single_batch(processed: BoltzProcessedInput, mol_dir: Path, device: torch.device) -> dict:
    dm = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )
    batch = next(iter(dm.predict_dataloader()))
    skip = {
        "all_coords",
        "all_resolved_mask",
        "crop_to_all_atom_map",
        "chain_symmetries",
        "amino_acids_symmetries",
        "ligand_symmetries",
        "record",
        "affinity_mw",
    }
    out = {}
    for k, v in batch.items():
        out[k] = v if k in skip else v.to(device)
    return out


def expand_feats_to_batch(feats: dict, batch_size: int) -> dict:
    if batch_size == 1:
        return feats
    out: dict = {}
    for k, v in feats.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
            out[k] = v.repeat(batch_size, *([1] * (v.dim() - 1)))
        else:
            out[k] = v
    return out


def load_model(cache: Path, device: torch.device) -> Boltz2:
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(use_paired_feature=True)
    diffusion_params = Boltz2DiffusionParams()
    steering_args = BoltzSteeringParams()

    predict_args = {
        "recycling_steps": 1,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    model = Boltz2.load_from_checkpoint(
        cache / "boltz2_conf.ckpt",
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=True,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    return model.to(device)


def pairformer_trunk_only(
    model: Boltz2,
    feats: dict[str, Tensor],
    recycling_steps: int,
    s_seed: Tensor,
    z_seed: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run trunk/pairformer to produce (s,z) without diffusion sampling."""
    s_inputs = model.input_embedder(feats)
    s_init = model.s_init(s_inputs)
    z_init = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
    relative_position_encoding = model.rel_pos(feats)
    z_init = z_init + relative_position_encoding
    z_init = z_init + model.token_bonds(feats["token_bonds"].float())
    if model.bond_type_feature:
        z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
    z_init = z_init + model.contact_conditioning(feats)

    s = s_seed
    z = z_seed
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    for _ in range(recycling_steps + 1):
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z_init + model.z_recycle(model.z_norm(z))
        if model.use_templates:
            # in this script we rely on processed input (no template yaml), so templates should be absent
            pass
        # keep MSA module behavior consistent with checkpoint setting
        z = z + model.msa_module(z, s_inputs, feats, use_kernels=model.use_kernels)
        s, z = model.pairformer_module(s, z, mask=mask, pair_mask=pair_mask, use_kernels=model.use_kernels)
    return s, z


def denoise_at_timestep(
    model: Boltz2,
    feats: dict[str, Tensor],
    s_inputs: Tensor,
    s_trunk: Tensor,
    z_trunk: Tensor,
    relative_position_encoding: Tensor,
    atom_coords_noisy: Tensor,
    t_hat: float,
    multiplicity: int = 1,
) -> Tensor:
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = model.diffusion_conditioning(
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        relative_position_encoding=relative_position_encoding,
        feats=feats,
    )
    diffusion_cond = {
        "q": q,
        "c": c,
        "to_keys": to_keys,
        "atom_enc_bias": atom_enc_bias,
        "atom_dec_bias": atom_dec_bias,
        "token_trans_bias": token_trans_bias,
    }
    return model.structure_module.preconditioned_network_forward(
        atom_coords_noisy,
        t_hat,
        network_condition_kwargs={
            "s_inputs": s_inputs,
            "s_trunk": s_trunk,
            "feats": feats,
            "diffusion_conditioning": diffusion_cond,
            "multiplicity": multiplicity,
        },
    )


def grad_norm(loss_term: Tensor, param: Tensor) -> float:
    grad = torch.autograd.grad(loss_term, param, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        return 0.0
    return float(grad.norm().item())


def write_structure_from_coords(processed: BoltzProcessedInput, batch: dict, coords: Tensor, out_cif: Path) -> None:
    record = batch["record"][0]
    atom_pad_mask = batch["atom_pad_mask"][0]
    structure = StructureV2.load(processed.targets_dir / f"{record.id}.npz").remove_invalid_chains()

    coord_unpad = coords[atom_pad_mask.bool()].detach().cpu().numpy()
    atoms = structure.atoms.copy()
    atoms["coords"] = coord_unpad
    atoms["is_present"] = True
    residues = structure.residues.copy()
    residues["is_present"] = True
    coord_np = np.array([(x,) for x in coord_unpad], dtype=Coords)

    new_structure = StructureV2(
        atoms=atoms,
        residues=residues,
        chains=structure.chains,
        bonds=structure.bonds,
        interfaces=np.array([], dtype=Interface),
        mask=structure.mask,
        coords=coord_np,
        ensemble=structure.ensemble,
        pocket=getattr(structure, "pocket", None),
    )
    out_cif.parent.mkdir(parents=True, exist_ok=True)
    with out_cif.open("w") as f:
        f.write(to_mmcif(new_structure, boltz2=True))


def parse_args() -> OptimConfig:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sequence",
        action="append",
        required=True,
        help="Protein chain sequence. Provide twice for a 2-chain complex (repeat flag).",
    )
    p.add_argument(
        "--chain_id",
        action="append",
        default=None,
        help="Chain IDs matching --sequence order (repeat flag). Default: A, B, C...",
    )
    p.add_argument("--pdb_id", required=True)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--cache", default=None, type=Path)

    p.add_argument("--recycling_steps", type=int, default=3)
    p.add_argument("--sampling_steps", type=int, default=200)
    p.add_argument("--diffusion_samples", type=int, default=1)

    p.add_argument("--outer_steps", type=int, default=5)
    p.add_argument("--inner_steps", type=int, default=1)
    p.add_argument("--stop_opt_after_t", type=int, default=160)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--w_iptm", type=float, default=0.8)
    p.add_argument("--w_ptm", type=float, default=0.2)
    p.add_argument(
        "--optimize_s",
        action="store_true",
        help="Also optimize s_opt (default is z-only since s grads may be missing).",
    )

    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="boltz-embedding-iptm-optim")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)

    args = p.parse_args()
    sequences = list(args.sequence)
    if len(sequences) < 1:
        raise ValueError("At least one --sequence is required.")
    if args.chain_id is None:
        chain_ids = [chr(ord("A") + i) for i in range(len(sequences))]
    else:
        chain_ids = list(args.chain_id)
        if len(chain_ids) != len(sequences):
            raise ValueError(
                f"Got {len(sequences)} sequences but {len(chain_ids)} chain_ids; they must match."
            )
    return OptimConfig(
        sequences=sequences,
        pdb_id=args.pdb_id,
        chain_ids=chain_ids,
        out_dir=args.out_dir,
        cache=(args.cache if args.cache is not None else Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()),
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
        outer_steps=args.outer_steps,
        inner_steps=args.inner_steps,
        stop_opt_after_t=args.stop_opt_after_t,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        w_iptm=args.w_iptm,
        w_ptm=args.w_ptm,
        optimize_s=args.optimize_s,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )


def main() -> int:  # noqa: C901
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "work").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    wandb_run = None
    if cfg.use_wandb:
        try:
            wandb = importlib.import_module("wandb")
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                config={k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("W&B init failed: %s", exc)
            wandb_run = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.cache, device)
    mol_dir = cfg.cache / "mols"

    if not hasattr(model, "confidence_module") or model.confidence_module is None:
        raise RuntimeError("Loaded model has no confidence_module; cannot compute ptm/iptm objective.")
    if not hasattr(model, "distogram_module") or model.distogram_module is None:
        raise RuntimeError("Loaded model has no distogram_module; cannot compute ptm/iptm objective.")

    # Prepare input (MSA, no template)
    work = cfg.out_dir / "work"
    yaml_path = work / "input.yaml"
    create_yaml(cfg.sequences, cfg.chain_ids, yaml_path, template_cif=None, msa=None)
    processed = process_single_input(
        input_path=yaml_path,
        out_dir=work / "processed_internal",
        cache=cfg.cache,
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
    )
    batch = load_single_batch(processed, mol_dir=mol_dir, device=device)
    feats = expand_feats_to_batch(batch, cfg.batch_size)

    with torch.no_grad():
        s_inputs = model.input_embedder(feats)
        relative_position_encoding = model.rel_pos(feats)

    # Baseline embeddings from pairformer (zero seed -> run trunk once)
    with torch.no_grad():
        n_tok = feats["token_pad_mask"].shape[1]
        s0_single = torch.zeros((1, n_tok, model.hparams.token_s), device=device)
        z0_single = torch.zeros((1, n_tok, n_tok, model.hparams.token_z), device=device)
        feats_single = {k: (v[:1] if torch.is_tensor(v) and v.dim() >= 1 else v) for k, v in feats.items()}
        s_base_single, z_base_single = pairformer_trunk_only(
            model=model,
            feats=feats_single,
            recycling_steps=cfg.recycling_steps,
            s_seed=s0_single,
            z_seed=z0_single,
        )
        s_base = s_base_single.repeat(cfg.batch_size, 1, 1).detach()
        z_base = z_base_single.repeat(cfg.batch_size, 1, 1, 1).detach()

    # Trainable embeddings
    s_opt = s_base.clone().detach().requires_grad_(True)
    z_opt = z_base.clone().detach().requires_grad_(True)

    atom_pad_mask = feats["atom_pad_mask"]
    atom_shape = (*atom_pad_mask.shape, 3)
    s_inputs_float = s_inputs.float()

    structure_mod = model.structure_module
    sigmas = structure_mod.sample_schedule(cfg.sampling_steps)
    gammas = torch.where(sigmas > structure_mod.gamma_min, structure_mod.gamma_0, 0.0)
    sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
    step_scale = structure_mod.step_scale
    noise_scale = structure_mod.noise_scale
    alignment_reverse_diff = structure_mod.alignment_reverse_diff
    num_diffusion_steps = len(sigmas_and_gammas)

    global_step = 0
    LOG.info(
        "Starting iPTM optimization: outer=%d diffusion_steps=%d inner=%d stop_opt_after_t=%d",
        cfg.outer_steps,
        num_diffusion_steps,
        cfg.inner_steps,
        cfg.stop_opt_after_t,
    )
    loss_history = []

    for outer in range(cfg.outer_steps):
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(atom_shape, device=device)
        loss_iteration_dict = {"loss": [], "ptm": [], "iptm": []}

        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            sigma_tm_val = sigma_tm.item()
            sigma_t_val = sigma_t.item()
            gamma_val = gamma.item()

            t_hat = sigma_tm_val * (1 + gamma_val)
            noise_var = noise_scale**2 * (t_hat**2 - sigma_tm_val**2)
            eps = (
                sqrt(noise_var) * torch.randn(atom_shape, device=device)
                if noise_var > 0
                else torch.zeros(atom_shape, device=device)
            )
            atom_coords_noisy = (atom_coords + eps).detach()

            do_opt = step_idx < cfg.stop_opt_after_t
            if do_opt:
                params = [z_opt] if not cfg.optimize_s else [s_opt, z_opt]
                optimizer = torch.optim.Adam(params, lr=cfg.lr)

            for inner in range(cfg.inner_steps if do_opt else 0):
                optimizer.zero_grad(set_to_none=True)

                s_trunk = s_opt.float()
                z_trunk = z_opt.float()

                x_0_hat = denoise_at_timestep(
                    model=model,
                    feats=feats,
                    s_inputs=s_inputs_float,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    relative_position_encoding=relative_position_encoding,
                    atom_coords_noisy=atom_coords_noisy,
                    t_hat=t_hat,
                    multiplicity=1,
                )

                # Compute distogram logits from z (same as model forward path)
                pdistogram = model.distogram_module(z_trunk)
                pred_distogram_logits = pdistogram[:, :, :, 0]  # (B, L, L, bins)

                # Confidence module returns ptm / iptm among other metrics.
                conf = model.confidence_module(
                    s_inputs=s_inputs_float,
                    s=s_trunk,
                    z=z_trunk,
                    x_pred=x_0_hat,
                    feats=feats,
                    pred_distogram_logits=pred_distogram_logits,
                    multiplicity=1,
                    run_sequentially=False,
                    use_kernels=model.use_kernels,
                )
                ptm = conf["ptm"]
                iptm = conf["iptm"]
                objective = cfg.w_iptm * iptm + cfg.w_ptm * ptm
                # Maximize objective -> minimize negative objective.
                loss = -objective.mean()

                # Log the losses
                loss_iteration_dict["loss"].append(float(loss.item()))
                loss_iteration_dict["ptm"].append(float(ptm.mean().item()))
                loss_iteration_dict["iptm"].append(float(iptm.mean().item()))

                # Gradient attribution (optional but useful)
                iptm_s_grad = grad_norm(iptm.mean(), s_opt)
                iptm_z_grad = grad_norm(iptm.mean(), z_opt)
                ptm_s_grad = grad_norm(ptm.mean(), s_opt)
                ptm_z_grad = grad_norm(ptm.mean(), z_opt)

                loss.backward()
                torch.nn.utils.clip_grad_norm_([z_opt], cfg.max_grad_norm)
                if cfg.optimize_s:
                    torch.nn.utils.clip_grad_norm_([s_opt], cfg.max_grad_norm)
                optimizer.step()

                payload = {
                    "outer": outer,
                    "diffusion_step": step_idx,
                    "inner": inner,
                    "global_step": global_step,
                    "t_hat": t_hat,
                    "ptm": float(ptm.mean().item()),
                    "iptm": float(iptm.mean().item()),
                    "objective": float(objective.mean().item()),
                    "loss": float(loss.item()),
                    "s_opt_norm": float(s_opt.detach().norm().item()),
                    "z_opt_norm": float(z_opt.detach().norm().item()),
                    "grads/iptm_s_grad_norm": iptm_s_grad,
                    "grads/iptm_z_grad_norm": iptm_z_grad,
                    "grads/ptm_s_grad_norm": ptm_s_grad,
                    "grads/ptm_z_grad_norm": ptm_z_grad,
                }
                if wandb_run is not None:
                    wandb_run.log(payload)
                if inner == cfg.inner_steps - 1:
                    LOG.info(
                        "outer=%d step=%d/%d inner=%d iptm=%.4f ptm=%.4f obj=%.4f",
                        outer,
                        step_idx,
                        num_diffusion_steps,
                        inner,
                        float(iptm.mean().item()),
                        float(ptm.mean().item()),
                        float(objective.mean().item()),
                    )
                global_step += 1

            # Denoising step forward (always)
            with torch.no_grad():
                s_trunk_detached = s_opt.float()
                z_trunk_detached = z_opt.float()
                atom_coords_denoised = denoise_at_timestep(
                    model=model,
                    feats=feats,
                    s_inputs=s_inputs_float,
                    s_trunk=s_trunk_detached,
                    z_trunk=z_trunk_detached,
                    relative_position_encoding=relative_position_encoding,
                    atom_coords_noisy=atom_coords_noisy,
                    t_hat=t_hat,
                    multiplicity=1,
                )
                if alignment_reverse_diff:
                    atom_mask_float = atom_pad_mask.float()
                    with torch.autocast("cuda", enabled=False):
                        atom_coords_noisy = weighted_rigid_align(
                            atom_coords_noisy.float(),
                            atom_coords_denoised.float(),
                            atom_mask_float,
                            atom_mask_float,
                        )
                    atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)
                denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
                atom_coords = atom_coords_noisy + step_scale * (sigma_t_val - t_hat) * denoised_over_sigma

        # Save final structure and embeddings for this outer loop
        out_cif = cfg.out_dir / f"optimized_outer{outer}.cif"
        try:
            write_structure_from_coords(processed, batch, atom_coords[0], out_cif)
            LOG.info("Saved final structure to %s", out_cif)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Failed to write final structure: %s", exc)

        torch.save(
            {
                "s_base": s_base.detach().cpu(),
                "z_base": z_base.detach().cpu(),
                "s_opt": s_opt.detach().cpu(),
                "z_opt": z_opt.detach().cpu(),
            },
            cfg.out_dir / f"embeddings_outer{outer}.pt",
        )
        loss_history.append(loss_iteration_dict)
        torch.save(loss_history, "losses_iptm_boltz_checkpoint.pt")

    torch.save(loss_history, "losses_iptm_boltz_checkpoint.pt")
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


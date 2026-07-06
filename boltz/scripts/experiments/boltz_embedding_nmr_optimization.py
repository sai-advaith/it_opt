#!/usr/bin/env python3
"""Boltz-2 embedding-space optimization with NMR loss (post-hoc / guided sampling).

Same 3-level optimization structure used by the other IT-opt drivers in this
repo (outer noise realizations -> diffusion timesteps -> inner embedding steps).

Architecture:
  Outer loop  – new diffusion noise realizations (fresh x_T each time).
  Middle loop – walk through each diffusion timestep t.
  Inner loop  – at each timestep, optimise (s_opt, z_opt) by denoising
                the *same* noisy structure at *that* noise level, computing
                loss on x_0_hat, backpropping into s_opt/z_opt, and
                updating with Adam.
  After inner loop – take the actual Euler diffusion step forward with the
                     updated embeddings.

Loss stack:
  main_loss       = NMR restraint loss
  bond_length_loss= covalent geometry + collision penalty
  violation_loss  = OpenFold-style structural violations
  anchor_loss     = L2 regularization toward the pairformer baseline
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from biotite.structure.io.pdb import PDBFile
import biotite.structure.io.pdb as biotite_pdb

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.parse.mmcif import parse_mmcif
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
from boltz.model.models.boltz2 import Boltz2
from boltz.model.modules.utils import compute_random_augmentation
from boltz.model.loss.diffusionv2 import weighted_rigid_align
from boltz.experiments.nmr.assets import NMRAssets, resolve_assets_dir
from boltz.experiments.nmr.nmr_loss_function import NMRLossFunction
from boltz.experiments.nmr.bond_length_loss_function import BondLengthLossFunction
from boltz.experiments.nmr.violation_loss_function import ViolationLossFunction
from boltz.experiments.nmr.anchor_loss_function import AnchorLossFunction

LOG = logging.getLogger("boltz_embedding_nmr_optimization")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class OptimConfig:
    sequence: str
    pdb_id: str
    chain_id: str = "A"
    out_dir: Path = Path("./boltz_embedding_nmr_optim")
    cache: Path = Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()

    # Boltz sampling / diffusion
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    use_template_module: bool = False
    use_msa_module: bool = True
    use_msa_server: bool = True
    msa_server_url: str = "https://api.colabfold.com"

    # Optimization loop (mirrors msa_batch_optimization_manager)
    outer_steps: int = 1          # number of fresh diffusion noise realizations
    inner_steps: int = 10         # optimization steps at each diffusion timestep
    stop_opt_after_t: int = 150   # freeze embeddings for step_idx >= this value
    lr: float = 0.09
    max_grad_norm: float = 1.0
    batch_size: int = 1

    # Loss weights
    lambda_violation: float = 1.0
    lambda_bond_length: float = 1.0
    lambda_anchor_s: float = 0.0
    lambda_anchor_z: float = 0.0

    # NMR assets (vendored pickles). Defaults to the copy shipped inside the
    # boltz package; override with --nmr_assets_dir or $BOLTZ_NMR_ASSETS_DIR.
    nmr_assets_dir: Path = field(default_factory=resolve_assets_dir)
    nmr_restraint_file: Path | None = None
    nmr_reference_pdb: Path | None = None
    nmr_methyl_relax_file: Path | None = None
    nmr_amide_rdc_file: Path | None = None
    nmr_amide_relax_file: Path | None = None
    nmr_methyl_rdc_file: Path | None = None

    methyl_relax_scale: float = 0.0
    methyl_rdc_scale: float = 0.0
    amide_rdc_scale: float = 0.0
    amide_relax_scale: float = 0.0
    noe_scale: float = 1.0
    iid_loss: bool = False

    atom_index_map_npy: Path | None = None

    # W&B
    use_wandb: bool = True
    wandb_project: str = "boltz-embedding-nmr-optim"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers – Boltz I/O
# ---------------------------------------------------------------------------
def create_yaml(seq: str, chain_id: str, path: Path, template_cif: Path | None, msa: str | None) -> None:
    data: dict[str, Any] = {
        "version": 1,
        "sequences": [{"protein": {"id": chain_id, "sequence": seq}}],
    }
    if msa is not None:
        data["sequences"][0]["protein"]["msa"] = msa
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


# ---------------------------------------------------------------------------
# Pairformer-only forward (no diffusion) – computes baseline (s_base, z_base)
# ---------------------------------------------------------------------------
def pairformer_trunk_only(
    model: Boltz2,
    feats: dict[str, Tensor],
    recycling_steps: int,
    s_seed: Tensor,
    z_seed: Tensor,
    use_template_module: bool,
    use_msa_module: bool,
) -> tuple[Tensor, Tensor]:
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
        if model.use_templates and use_template_module:
            z = z + model.template_module(z, feats, pair_mask, use_kernels=model.use_kernels)
        if use_msa_module:
            z = z + model.msa_module(z, s_inputs, feats, use_kernels=model.use_kernels)
        s, z = model.pairformer_module(
            s, z, mask=mask, pair_mask=pair_mask, use_kernels=model.use_kernels
        )
    return s, z


# ---------------------------------------------------------------------------
# Single-timestep denoising (differentiable w.r.t. s_trunk, z_trunk)
# ---------------------------------------------------------------------------
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
    """Denoise noisy atom coords at a single diffusion timestep.

    Returns x_0_hat (predicted clean coords).  The computation graph is
    connected back to s_trunk and z_trunk, so gradients flow.
    """
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
        model.diffusion_conditioning(
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            feats=feats,
        )
    )
    diffusion_cond = {
        "q": q,
        "c": c,
        "to_keys": to_keys,
        "atom_enc_bias": atom_enc_bias,
        "atom_dec_bias": atom_dec_bias,
        "token_trans_bias": token_trans_bias,
    }

    x_0_hat = model.structure_module.preconditioned_network_forward(
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
    return x_0_hat


def grad_norm(loss_term: Tensor, param: Tensor) -> float:
    """Return ||d loss_term / d param|| with safe fallback."""
    grad = torch.autograd.grad(
        loss_term,
        param,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return 0.0
    return float(grad.norm().item())


# ---------------------------------------------------------------------------
# Structure writing
# ---------------------------------------------------------------------------
def write_structure_from_coords(processed: BoltzProcessedInput, batch: dict, coords: Tensor, out_cif: Path) -> None:
    record = batch["record"][0]
    atom_pad_mask = batch["atom_pad_mask"][0]
    structure = StructureV2.load(processed.targets_dir / f"{record.id}.npz")
    structure = structure.remove_invalid_chains()

    coord_unpad = coords[atom_pad_mask.bool()].detach().cpu().numpy()
    atoms = structure.atoms.copy()
    atoms["coords"] = coord_unpad
    atoms["is_present"] = True
    residues = structure.residues.copy()
    residues["is_present"] = True
    coord_np = np.array([(x,) for x in coord_unpad], dtype=Coords)
    # StructureV2 requires `mask` and `ensemble` fields. Keep them consistent with
    # the loaded target topology and update only coordinates-related tables.
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> OptimConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--pdb_id", required=True)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--cache", default=None, type=Path)

    p.add_argument("--recycling_steps", type=int, default=3)
    p.add_argument("--sampling_steps", type=int, default=200)
    p.add_argument("--diffusion_samples", type=int, default=1)

    p.add_argument("--outer_steps", type=int, default=20)
    p.add_argument("--inner_steps", type=int, default=1)
    p.add_argument(
        "--stop_opt_after_t",
        type=int,
        default=160,
        help="Freeze embeddings (skip inner optimization) for diffusion step_idx >= this value.",
    )
    p.add_argument("--lr", type=float, default=0.025)
    p.add_argument("--max_grad_norm", type=float, default=0.6)
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--lambda_violation", type=float, default=0.05)
    p.add_argument("--lambda_bond_length", type=float, default=0.01)
    p.add_argument("--lambda_anchor_s", type=float, default=1e-3)
    p.add_argument("--lambda_anchor_z", type=float, default=1e-3)

    p.add_argument("--nmr_restraint_file", type=Path, required=True)
    p.add_argument("--nmr_reference_pdb", type=Path, required=True)
    p.add_argument("--nmr_assets_dir", type=Path, default=None)
    p.add_argument("--atom_index_map_npy", type=Path, default=None)

    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="boltz-embedding-nmr-optim")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)

    args = p.parse_args()
    cfg = OptimConfig(
        sequence=args.sequence,
        pdb_id=args.pdb_id,
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
        lambda_violation=args.lambda_violation,
        lambda_bond_length=args.lambda_bond_length,
        lambda_anchor_s=args.lambda_anchor_s,
        lambda_anchor_z=args.lambda_anchor_z,
        nmr_restraint_file=args.nmr_restraint_file,
        nmr_reference_pdb=args.nmr_reference_pdb,
        atom_index_map_npy=args.atom_index_map_npy,
        nmr_assets_dir=(args.nmr_assets_dir if args.nmr_assets_dir is not None else resolve_assets_dir()),
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    # ---- W&B ----
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
        except Exception as exc:
            LOG.warning("W&B init failed: %s", exc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.cache, device)
    mol_dir = cfg.cache / "mols"

    # ---- Prepare Boltz input (MSA, no template) ----
    work = cfg.out_dir / "work"
    yaml_path = work / "input.yaml"
    create_yaml(cfg.sequence, cfg.chain_id, yaml_path, template_cif=None, msa=None)
    processed = process_single_input(
        input_path=yaml_path,
        out_dir=work / "processed_internal",
        cache=cfg.cache,
        use_msa_server=cfg.use_msa_server,
        msa_server_url=cfg.msa_server_url,
    )
    batch = load_single_batch(processed, mol_dir=mol_dir, device=device)
    feats = expand_feats_to_batch(batch, cfg.batch_size)

    # ---- Pre-compute model quantities that don't depend on trainable embeddings ----
    with torch.no_grad():
        s_inputs = model.input_embedder(feats)
        relative_position_encoding = model.rel_pos(feats)

    # ---- NMR loss ----
    assets = NMRAssets(cfg.nmr_assets_dir)
    assets.validate()
    pdb_file = PDBFile.read(str(cfg.nmr_reference_pdb))
    atom_array = biotite_pdb.get_structure(pdb_file, model=1, include_bonds=True)
    # Strip hydrogens/deuterium and OXT: Boltz does not predict these atoms,
    # so topology must match the heavy-atom set Boltz produces.
    atom_array = atom_array[
        (atom_array.element != "H")
        & (atom_array.element != "D")
        & (atom_array.atom_name != "OXT")
    ]

    nmr_loss_fn = NMRLossFunction(
        restraint_file=str(cfg.nmr_restraint_file),
        pdb_file=str(cfg.nmr_reference_pdb),
        atom_array=atom_array,
        device=str(device),
        iid_loss=cfg.iid_loss,
        methyl_relax_file=str(cfg.nmr_methyl_relax_file) if cfg.nmr_methyl_relax_file else None,
        methyl_relax_scale=cfg.methyl_relax_scale,
        amide_rdc_file=str(cfg.nmr_amide_rdc_file) if cfg.nmr_amide_rdc_file else None,
        amide_rdc_scale=cfg.amide_rdc_scale,
        amide_relax_file=str(cfg.nmr_amide_relax_file) if cfg.nmr_amide_relax_file else None,
        amide_relax_scale=cfg.amide_relax_scale,
        methyl_rdc_file=str(cfg.nmr_methyl_rdc_file) if cfg.nmr_methyl_rdc_file else None,
        methyl_rdc_scale=cfg.methyl_rdc_scale,
        noe_scale=cfg.noe_scale,
        fragments_pickle=str(assets.fragments_pickle),
        names_pickle=str(assets.names_pickle),
    )

    # ---- Auxiliary losses (bond length, violation, anchor) ----
    bond_length_loss_fn = BondLengthLossFunction(atom_array, cfg.lambda_bond_length, str(device))
    violation_loss_fn = ViolationLossFunction(atom_array, cfg.lambda_violation)

    # ---- Compute pairformer baseline (s_base, z_base) from zero seed ----
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
            use_template_module=cfg.use_template_module,
            use_msa_module=cfg.use_msa_module,
        )
        s_base = s_base_single.repeat(cfg.batch_size, 1, 1).detach()
        z_base = z_base_single.repeat(cfg.batch_size, 1, 1, 1).detach()

    # ---- Anchor loss (regularises toward baseline) ----
    init_s=s_base_single.squeeze(0).detach()
    init_z=z_base_single.squeeze(0).detach()
    anchor_loss_fn = AnchorLossFunction(
        init_s=init_s,
        init_z=init_z,
        pdb_id=cfg.pdb_id,
        lambda_s=cfg.lambda_anchor_s,
        lambda_z=cfg.lambda_anchor_z,
        device=str(device),
    )

    # ---- Atom index map (optional: maps Boltz output atoms → NMR atom ordering) ----
    atom_index_map: Tensor | None = None
    if cfg.atom_index_map_npy is not None:
        atom_index_map = torch.from_numpy(np.load(cfg.atom_index_map_npy)).long().to(device)

    atom_pad_mask = feats["atom_pad_mask"]  # (B, N_atom)

    # ---- Pre-compute s_inputs for the batch (doesn't depend on delta) ----
    s_inputs_float = s_inputs.float()

    # ---- Diffusion schedule (from the model's AtomDiffusion module) ----
    structure_mod = model.structure_module
    sigmas = structure_mod.sample_schedule(cfg.sampling_steps)
    gammas = torch.where(sigmas > structure_mod.gamma_min, structure_mod.gamma_0, 0.0)
    sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
    step_scale = structure_mod.step_scale
    noise_scale = structure_mod.noise_scale
    alignment_reverse_diff = structure_mod.alignment_reverse_diff
    num_diffusion_steps = len(sigmas_and_gammas)

    # ---- Optimisable embeddings (direct optimization, no delta parameterization) ----
    s_opt = s_base.clone().detach().requires_grad_(True)
    z_opt = z_base.clone().detach().requires_grad_(True)

    # ---- Atom shape for noise ----
    atom_shape = (*atom_pad_mask.shape, 3)  # (B, N_atom, 3)

    global_step = 0
    LOG.info(
        "Starting 3-level optimization: outer=%d, diffusion_steps=%d, inner=%d",
        cfg.outer_steps, num_diffusion_steps, cfg.inner_steps,
    )

    # Add a loss dict
    losses_it_opt = []

    # =======================================================================
    # Outer loop: fresh diffusion noise realizations
    # =======================================================================
    for outer in range(cfg.outer_steps):
        init_sigma = sigmas[0]
        loss_dict_iteration = {"nmr_loss": [], "ub_constraints": [], "lb_constraints": []}
        atom_coords = init_sigma * torch.randn(atom_shape, device=device)
        atom_coords_denoised: Tensor | None = None

        LOG.info("=== Outer loop %d/%d ===", outer + 1, cfg.outer_steps)

        # ===================================================================
        # Middle loop: walk through diffusion timesteps
        # ===================================================================
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            sigma_tm_val = sigma_tm.item()
            sigma_t_val = sigma_t.item()
            gamma_val = gamma.item()

            t_hat = sigma_tm_val * (1 + gamma_val)
            noise_var = noise_scale**2 * (t_hat**2 - sigma_tm_val**2)
            eps = sqrt(noise_var) * torch.randn(atom_shape, device=device) if noise_var > 0 else torch.zeros(atom_shape, device=device)
            atom_coords_noisy = (atom_coords + eps).detach()

            # Freeze embeddings after a chosen diffusion step to reduce distributional shift.
            do_opt = step_idx < cfg.stop_opt_after_t
            if do_opt:
                # Re-create optimizer each diffusion step (matches reference)
                optimizer = torch.optim.Adam([s_opt, z_opt], lr=cfg.lr)

            # ===============================================================
            # Inner loop: optimise s_opt, z_opt at this noise level
            # ===============================================================
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

                # Unpad coords → shape expected by NMR loss
                x_0_hat_unpad = x_0_hat[:, atom_pad_mask[0].bool(), :]
                coords_for_loss = x_0_hat_unpad
                if atom_index_map is not None:
                    coords_for_loss = x_0_hat_unpad[:, atom_index_map, :]

                # Losses
                main_loss, _ = nmr_loss_fn(coords_for_loss, time=step_idx)
                bond_loss, _ = bond_length_loss_fn(coords_for_loss, None)
                viol_loss, _ = violation_loss_fn(coords_for_loss, None)

                anchor_result, _ = anchor_loss_fn(
                    coords_for_loss, None, msa=(s_trunk, z_trunk)
                )
                anchor_s, anchor_z = anchor_result

                # Per-component gradient attribution for embeddings.
                nmr_s_grad_norm = grad_norm(main_loss, s_opt)
                nmr_z_grad_norm = grad_norm(main_loss, z_opt)
                bond_s_grad_norm = grad_norm(bond_loss, s_opt)
                bond_z_grad_norm = grad_norm(bond_loss, z_opt)
                viol_s_grad_norm = grad_norm(viol_loss, s_opt)
                viol_z_grad_norm = grad_norm(viol_loss, z_opt)
                anchor_s_on_s_grad_norm = grad_norm(anchor_s, s_opt)
                anchor_s_on_z_grad_norm = grad_norm(anchor_s, z_opt)
                anchor_z_on_s_grad_norm = grad_norm(anchor_z, s_opt)
                anchor_z_on_z_grad_norm = grad_norm(anchor_z, z_opt)

                total_loss = main_loss + bond_loss + viol_loss + anchor_s + anchor_z
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_([s_opt], cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_([z_opt], cfg.max_grad_norm)
                optimizer.step()

                # Logging
                log_payload: dict[str, Any] = {
                    "outer": outer,
                    "diffusion_step": step_idx,
                    "inner": inner,
                    "global_step": global_step,
                    "t_hat": t_hat,
                    "total_loss": float(total_loss.item()),
                    "nmr_loss": float(main_loss.item()),
                    "bond_length_loss": float(bond_loss.item()) if torch.is_tensor(bond_loss) else float(bond_loss),
                    "violation_loss": float(viol_loss.item()) if torch.is_tensor(viol_loss) else float(viol_loss),
                    "anchor_loss_s": float(anchor_s.item()) if torch.is_tensor(anchor_s) else float(anchor_s),
                    "anchor_loss_z": float(anchor_z.item()) if torch.is_tensor(anchor_z) else float(anchor_z),
                    "s_opt_norm": float(s_opt.detach().norm().item()),
                    "z_opt_norm": float(z_opt.detach().norm().item()),
                    "grads/nmr_s_grad_norm": nmr_s_grad_norm,
                    "grads/nmr_z_grad_norm": nmr_z_grad_norm,
                    "grads/bond_s_grad_norm": bond_s_grad_norm,
                    "grads/bond_z_grad_norm": bond_z_grad_norm,
                    "grads/viol_s_grad_norm": viol_s_grad_norm,
                    "grads/viol_z_grad_norm": viol_z_grad_norm,
                    "grads/anchor_s_on_s_grad_norm": anchor_s_on_s_grad_norm,
                    "grads/anchor_s_on_z_grad_norm": anchor_s_on_z_grad_norm,
                    "grads/anchor_z_on_s_grad_norm": anchor_z_on_s_grad_norm,
                    "grads/anchor_z_on_z_grad_norm": anchor_z_on_z_grad_norm,
                }
                if s_opt.grad is not None:
                    log_payload["s_opt_grad_norm"] = float(s_opt.grad.norm().item())
                if z_opt.grad is not None:
                    log_payload["z_opt_grad_norm"] = float(z_opt.grad.norm().item())

                if hasattr(nmr_loss_fn, "wandb_log"):
                    try:
                        for k, v in nmr_loss_fn.wandb_log(coords_for_loss).items():
                            log_payload[f"nmr/{k}"] = v
                    except Exception:
                        pass

                # loss_dict_iteration["loss_nmr"] = log_payload["nmr"]
                loss_dict_iteration["nmr_loss"].append(log_payload["nmr_loss"])
                loss_dict_iteration["ub_constraints"].append(log_payload["nmr/constraints_satisfied_ub"])
                loss_dict_iteration["lb_constraints"].append(log_payload["nmr/constraints_satisfied_lb"])

                if inner == cfg.inner_steps - 1:
                    LOG.info(
                        "outer=%d step=%d/%d inner=%d nmr=%.4f bond=%.4f viol=%.4f anchor_s=%.4f anchor_z=%.4f total=%.4f",
                        outer, step_idx, num_diffusion_steps, inner,
                        main_loss.item(),
                        float(bond_loss.item()) if torch.is_tensor(bond_loss) else float(bond_loss),
                        float(viol_loss.item()) if torch.is_tensor(viol_loss) else float(viol_loss),
                        float(anchor_s.item()) if torch.is_tensor(anchor_s) else float(anchor_s),
                        float(anchor_z.item()) if torch.is_tensor(anchor_z) else float(anchor_z),
                        total_loss.item(),
                    )

                if wandb_run is not None:
                    wandb_run.log(log_payload)
                global_step += 1

            if not do_opt and wandb_run is not None:
                wandb_run.log(
                    {
                        "outer": outer,
                        "diffusion_step": step_idx,
                        "global_step": global_step,
                        "opt_frozen": 1,
                    }
                )

            # ===============================================================
            # After inner loop: take the actual Euler diffusion step forward
            # ===============================================================
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

            # Periodic structure save during diffusion trajectory
            if step_idx > 0 and step_idx % max(1, num_diffusion_steps // 10) == 0:
                snap_path = cfg.out_dir / f"snapshot_outer{outer}_step{step_idx}.cif"
                try:
                    write_structure_from_coords(processed, batch, atom_coords[0], snap_path)
                except Exception:
                    pass

        # ===================================================================
        # End of diffusion trajectory – save final structure and embeddings
        # ===================================================================
        losses_it_opt.append(loss_dict_iteration)
        out_cif = cfg.out_dir / f"optimized_outer{outer}.cif"
        try:
            write_structure_from_coords(processed, batch, atom_coords[0], out_cif)
            LOG.info("Saved final structure to %s", out_cif)
        except Exception as e:
            LOG.warning("Failed to write final structure: %s", e)

        torch.save(
            {
                "s_base": s_base.detach().cpu(),
                "z_base": z_base.detach().cpu(),
                "s_opt": s_opt.detach().cpu(),
                "z_opt": z_opt.detach().cpu(),
            },
            cfg.out_dir / f"embeddings_outer{outer}.pt",
        )

        # Also save the full-batch final coords
        torch.save(atom_coords.detach().cpu(), cfg.out_dir / f"coords_outer{outer}.pt")
        
        torch.save(losses_it_opt, "losses_nmr_boltz_checkpoint.pt")

    torch.save(losses_it_opt, "losses_nmr_boltz_checkpoint.pt")
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

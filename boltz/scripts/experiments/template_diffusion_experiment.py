#!/usr/bin/env python3
"""
Template-Diffusion Relationship Experiment

This experiment explores the relationship between the template and diffusion modules in Boltz-2:

Round 1: Generate a structure using MSA (from server) but NO template.
         The diffusion module produces structure_0.

Round 2: For M iterations, use the previous structure as template, with NO MSA.
         Each iteration: template conditioning -> pairformer (N recycling steps) -> diffusion.
         After each iteration (except the first), compute RMSD between current and previous output.
         Log everything.

Usage:
    python scripts/experiments/template_diffusion_experiment.py \\
        --sequence "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ" \\
        --M 5 --N 3 --out_dir ./experiment_output
"""

from __future__ import annotations

import json
import importlib
import logging
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from torch import Tensor

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Coords, Interface, Manifest, StructureV2
from boltz.data.parse.mmcif import parse_mmcif
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
from boltz.model.loss.diffusionv2 import weighted_rigid_align

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for the template-diffusion experiment."""

    sequence: str
    M: int = 5  # Number of template-only iterations in round 2
    N: int = 3  # Recycling steps (pairformer passes) per iteration
    out_dir: Path = Path("./template_diffusion_experiment")
    cache: Path = Path("~/.boltz").expanduser()
    use_msa_server: bool = True
    msa_server_url: str = "https://api.colabfold.com"
    recycling_steps: int = 3  # Will be overridden by N for round 2
    sampling_steps: int = 200
    diffusion_samples: int = 1
    chain_id: str = "A"
    pdb_id: str = "1u0p"
    mode: Literal[
        "msa_then_template_no_msa",
        "msa_then_template_with_msa",
        "msa_then_recycle5_no_template",
        "msa_only_no_template",
    ] = "msa_then_template_no_msa"
    seed: int | None = 42
    use_wandb: bool = True
    wandb_project: str = "boltz-template-diffusion"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None


@dataclass
class ExperimentLog:
    """Log of the experiment for analysis."""

    config: dict
    round1: dict = field(default_factory=dict)
    round2_iterations: list[dict] = field(default_factory=list)
    rmsds: list[float] = field(default_factory=list)
    embedding_changes: list[dict] = field(default_factory=list)
    timestamps: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def backbone_atom_mask_from_structure(structure) -> np.ndarray:
    """Boolean mask over ``structure.atoms`` selecting protein backbone atoms.

    Ordering matches the flat ``structure.atoms`` array, so the returned mask
    can be indexed directly into a prediction coordinate tensor (after
    atom-pad removal).
    """
    backbone_names = {"N", "CA", "C", "O"}
    mask = np.zeros(len(structure.atoms), dtype=bool)
    for chain in structure.chains:
        if chain["mol_type"] != 0:  # 0 = PROTEIN
            continue
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        for res in structure.residues[res_start:res_end]:
            atom_start = res["atom_idx"]
            atom_end = res["atom_idx"] + res["atom_num"]
            for i in range(atom_start, atom_end):
                atom_name = str(structure.atoms[i]["name"]).strip()
                if atom_name in backbone_names:
                    mask[i] = True
    return mask


def ensemble_backbone_coords(
    sample_atom_coords: Tensor,
    atom_pad_mask: Tensor,
    backbone_mask: np.ndarray,
) -> torch.Tensor:
    """Return backbone coords per diffusion sample as ``(D, N_bb, 3)``.

    ``sample_atom_coords`` is the raw diffusion output of shape
    ``(D, N_atoms_padded, 3)`` where ``D`` is the number of diffusion samples.
    """
    keep = atom_pad_mask.bool()
    bb_idx = torch.from_numpy(np.where(backbone_mask)[0]).long().to(sample_atom_coords.device)
    out = []
    for d in range(sample_atom_coords.shape[0]):
        unpad = sample_atom_coords[d][keep]
        out.append(unpad.index_select(0, bb_idx))
    return torch.stack(out, dim=0).cpu()


def pairwise_structure_diff(
    curr_bb: torch.Tensor, prev_bb: torch.Tensor
) -> tuple[float, float]:
    """Compute Protenix-style ``(structure_diff, structure_diff_std)``.

    For each sample in ``curr_bb`` (shape ``(D_c, N, 3)``) compute RMSD to every
    sample in ``prev_bb`` (shape ``(D_p, N, 3)``) after optimal rigid alignment,
    take the min over the reference axis, then return mean and std over the
    ``D_c`` current samples. Matches the reduction Protenix uses in
    ``fixed_point_analysis.py``.
    """
    d_c = curr_bb.shape[0]
    d_p = prev_bb.shape[0]
    n = curr_bb.shape[1]
    if prev_bb.shape[1] != n:
        return float("nan"), float("nan")

    weights = torch.ones((1, n), dtype=torch.float32)
    mask = torch.ones((1, n), dtype=torch.float32)

    rmsd_matrix = torch.zeros((d_c, d_p), dtype=torch.float32)
    for i in range(d_c):
        for j in range(d_p):
            coords_ref = prev_bb[j].unsqueeze(0).float()
            coords_pred = curr_bb[i].unsqueeze(0).float()
            with torch.no_grad():
                aligned = weighted_rigid_align(coords_ref, coords_pred, weights, mask)
            mse = ((coords_pred - aligned) ** 2).sum(dim=-1)
            rmsd_matrix[i, j] = torch.sqrt(mse.mean(dim=-1)).item()

    per_sample = rmsd_matrix.min(dim=1).values  # (D_c,)
    mean = float(per_sample.mean().item())
    std = float(per_sample.std(unbiased=False).item()) if d_c > 1 else 0.0
    return mean, std


def extract_backbone_coords(structure) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract backbone atom coordinates and mask from a StructureV2.

    Returns coords (L, 3) and mask (L,) for backbone atoms (N, CA, C, O).
    """
    backbone_names = {"N", "CA", "C", "O"}
    coords_list = []
    mask_list = []

    for chain in structure.chains:
        if chain["mol_type"] != 0:  # 0 = PROTEIN
            continue
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        for res in structure.residues[res_start:res_end]:
            atom_start = res["atom_idx"]
            atom_end = res["atom_idx"] + res["atom_num"]
            for atom in structure.atoms[atom_start:atom_end]:
                atom_name = str(atom["name"]).strip()
                if atom_name in backbone_names and atom["is_present"]:
                    coords_list.append(atom["coords"])
                    mask_list.append(1.0)
    if not coords_list:
        raise ValueError("No backbone atoms found in structure")
    coords = torch.tensor(np.array(coords_list), dtype=torch.float32)
    mask = torch.tensor(np.array(mask_list), dtype=torch.float32)
    return coords, mask


def compute_rmsd(
    coords_ref: torch.Tensor,
    coords_pred: torch.Tensor,
    mask_ref: torch.Tensor,
    mask_pred: torch.Tensor,
) -> float:
    """
    Compute weighted RMSD after optimal superposition.

    Aligns ref to pred and computes RMSD. Both should have same number of atoms.
    """
    if coords_ref.shape != coords_pred.shape:
        raise ValueError(
            f"Coord shape mismatch: ref {coords_ref.shape} vs pred {coords_pred.shape}"
        )
    n = coords_ref.shape[0]
    weights = torch.ones(n, dtype=torch.float32)
    mask = (mask_ref > 0.5) & (mask_pred > 0.5)
    if mask.sum() < 4:
        return float("nan")
    coords_ref = coords_ref.unsqueeze(0)
    coords_pred = coords_pred.unsqueeze(0)
    mask = mask.unsqueeze(0)
    weights = weights.unsqueeze(0)
    with torch.no_grad():
        aligned = weighted_rigid_align(
            coords_ref.float(),
            coords_pred.float(),
            weights,
            mask,
        )
    mse = ((coords_pred - aligned) ** 2).sum(dim=-1) * mask
    rmsd = torch.sqrt(mse.sum() / mask.sum().clamp(min=1)).item()
    return float(rmsd)


def create_round1_yaml(seq: str, chain_id: str, path: Path) -> None:
    """Create YAML for round 1: MSA from server, no template."""
    data = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": chain_id,
                    "sequence": seq,
                }
            }
        ],
    }
    with path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def create_round2_yaml(
    seq: str,
    chain_id: str,
    template_path: Path | None,
    output_path: Path,
    msa_source: str = "empty",
) -> None:
    """Create YAML for round 2 with configurable template/MSA sources.

    ``msa_source`` controls what the MSA module sees:

    - ``"empty"``: single-sequence (query-only) dummy MSA — Boltz builds the
      1-row ``dummy_msa`` at featurization.
    - absolute path to a ``.csv`` or ``.a3m`` file: reuse that precomputed
      MSA (e.g. the CSV produced by round 1 when ``--with_msa`` modes want
      the real evolutionary signal carried into round 2).
    """
    data: dict = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": chain_id,
                    "sequence": seq,
                    "msa": msa_source,
                }
            }
        ],
    }
    if template_path is not None:
        data["templates"] = [{"cif": str(template_path.resolve())}]
    with output_path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def transfer_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move inference batch tensors to target device."""
    skip_keys = {
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
    for key, value in batch.items():
        if key in skip_keys:
            out[key] = value
        else:
            out[key] = value.to(device)
    return out


def process_single_input(
    input_path: Path,
    out_dir: Path,
    cache: Path,
    use_msa_server: bool,
    msa_server_url: str,
) -> BoltzProcessedInput:
    """Process one YAML input with Boltz internals (no CLI)."""
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
        constraints_dir=(
            processed_dir / "constraints"
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            processed_dir / "templates" if (processed_dir / "templates").exists() else None
        ),
        extra_mols_dir=(
            processed_dir / "mols" if (processed_dir / "mols").exists() else None
        ),
    )


def load_single_batch(
    processed: BoltzProcessedInput,
    mol_dir: Path,
    device: torch.device,
) -> dict:
    """Load one featurized inference batch."""
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )
    batch = next(iter(data_module.predict_dataloader()))
    return transfer_batch_to_device(batch, device)


def load_model(cache: Path, device: torch.device) -> Boltz2:
    """Load Boltz2 model module similarly to main.predict."""
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


def run_forward_with_seed(
    model: Boltz2,
    feats: dict[str, Tensor],
    recycling_steps: int,
    sampling_steps: int,
    diffusion_samples: int,
    init_state: tuple[Tensor, Tensor] | None = None,
    use_template_module: bool = True,
) -> dict[str, Tensor | list[dict[str, Tensor]]]:
    """Forward pass that can start pairformer from previous (s,z).

    The MSA module is always applied. When the input has no real MSA
    (``msa: "empty"``), the featurizer provides a single-sequence (query-only)
    MSA and the module runs on that, matching the training-time single-sequence
    regime rather than bypassing the module entirely.
    """
    with torch.no_grad():
        s_inputs = model.input_embedder(feats)
        s_init = model.s_init(s_inputs)
        z_init = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
        relative_position_encoding = model.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + model.token_bonds(feats["token_bonds"].float())
        if model.bond_type_feature:
            z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + model.contact_conditioning(feats)

        if init_state is None:
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)
        else:
            s_prev, z_prev = init_state
            if s_prev.shape != s_init.shape or z_prev.shape != z_init.shape:
                raise ValueError(
                    f"Seed shape mismatch: s {s_prev.shape} vs {s_init.shape}, "
                    f"z {z_prev.shape} vs {z_init.shape}"
                )
            s, z = s_prev.clone(), z_prev.clone()

        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        pairformer_trace: list[dict[str, Tensor]] = []

        for i in range(recycling_steps + 1):
            s = s_init + model.s_recycle(model.s_norm(s))
            z = z_init + model.z_recycle(model.z_norm(z))
            if model.use_templates and use_template_module:
                z = z + model.template_module(z, feats, pair_mask, use_kernels=model.use_kernels)
            # Always run the MSA module. With msa: "empty" the featurizer
            # provides a single-sequence MSA; skipping the module entirely is
            # off-distribution (the model was never trained with z += 0 here).
            z = z + model.msa_module(z, s_inputs, feats, use_kernels=model.use_kernels)
            s, z = model.pairformer_module(
                s,
                z,
                mask=mask,
                pair_mask=pair_mask,
                use_kernels=model.use_kernels,
            )
            pairformer_trace.append(
                {"recycle": torch.tensor(i), "s": s.detach().cpu(), "z": z.detach().cpu()}
            )

        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = model.diffusion_conditioning(
            s_trunk=s,
            z_trunk=z,
            relative_position_encoding=relative_position_encoding,
            feats=feats,
        )
        diffusion_conditioning = {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }
        struct_out = model.structure_module.sample(
            s_trunk=s.float(),
            s_inputs=s_inputs.float(),
            feats=feats,
            num_sampling_steps=sampling_steps,
            atom_mask=feats["atom_pad_mask"].float(),
            multiplicity=diffusion_samples,
            max_parallel_samples=1,
            steering_args=model.steering_args,
            diffusion_conditioning=diffusion_conditioning,
        )
        return {
            "s": s.detach(),
            "z": z.detach(),
            "sample_atom_coords": struct_out["sample_atom_coords"].detach(),
            "pairformer_trace": pairformer_trace,
        }


def write_prediction_structure(
    processed: BoltzProcessedInput,
    batch: dict,
    pred_coords: Tensor,
    out_cif: Path,
) -> StructureV2:
    """Write predicted coordinates to a CIF to reuse as template."""
    record = batch["record"][0]
    atom_pad_mask = batch["atom_pad_mask"][0]
    structure = StructureV2.load(processed.targets_dir / f"{record.id}.npz")
    structure = structure.remove_invalid_chains()

    coord_unpad = pred_coords[atom_pad_mask.bool()].cpu().numpy()
    atoms = structure.atoms.copy()
    atoms["coords"] = coord_unpad
    atoms["is_present"] = True
    residues = structure.residues.copy()
    residues["is_present"] = True
    coord_np = np.array([(x,) for x in coord_unpad], dtype=Coords)
    new_structure: StructureV2 = replace(
        structure,
        atoms=atoms,
        residues=residues,
        interfaces=np.array([], dtype=Interface),
        coords=coord_np,
    )
    out_cif.parent.mkdir(parents=True, exist_ok=True)
    with out_cif.open("w") as f:
        f.write(to_mmcif(new_structure, boltz2=True))
    return new_structure


def load_embeddings_for_structure(struct_path: Path) -> dict[str, np.ndarray] | None:
    """Load embeddings file corresponding to a predicted structure."""
    record_id = struct_path.parent.name
    emb_path = struct_path.parent / f"embeddings_{record_id}.npz"
    if not emb_path.exists():
        return None
    arr = np.load(emb_path)
    return {"s": arr["s"], "z": arr["z"]}


def compute_embedding_change(
    prev_embeddings: dict[str, np.ndarray], curr_embeddings: dict[str, np.ndarray]
) -> dict[str, float]:
    """Compute embedding drift metrics in the Protenix fixed-point style.

    Matches Protenix's ``fixed_point_analysis.py``:

    - ``s_diff = mean((curr - prev)^2) / ||prev||``   (Frobenius norm of prev)
    - ``z_diff = mean((curr - prev)^2) / ||prev||``
    - ``s_norm_diff = mean((normalize(curr, dim=-1) - normalize(prev, dim=-1))^2)``
    - ``z_norm_diff = same, per-pair L2 normalization along the feature dim``

    ``curr`` is this iteration's trunk output; ``prev`` is the previous
    iteration's trunk output (also the warm-start input to this iteration).
    """
    prev_s = prev_embeddings["s"]
    curr_s = curr_embeddings["s"]
    prev_z = prev_embeddings["z"]
    curr_z = curr_embeddings["z"]

    if prev_s.shape != curr_s.shape or prev_z.shape != curr_z.shape:
        raise ValueError(
            f"Embedding shape mismatch: s {prev_s.shape} vs {curr_s.shape}, "
            f"z {prev_z.shape} vs {curr_z.shape}"
        )

    eps = 1e-8

    def _l2_normalize_last(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(norm, eps, None)

    s_mse = float(np.mean(np.square(curr_s - prev_s)))
    z_mse = float(np.mean(np.square(curr_z - prev_z)))
    prev_s_fro = float(np.linalg.norm(prev_s))
    prev_z_fro = float(np.linalg.norm(prev_z))

    s_hat_curr = _l2_normalize_last(curr_s)
    s_hat_prev = _l2_normalize_last(prev_s)
    z_hat_curr = _l2_normalize_last(curr_z)
    z_hat_prev = _l2_normalize_last(prev_z)

    return {
        "s_diff": s_mse / (prev_s_fro + eps),
        "z_diff": z_mse / (prev_z_fro + eps),
        "s_norm_diff": float(np.mean(np.square(s_hat_curr - s_hat_prev))),
        "z_norm_diff": float(np.mean(np.square(z_hat_curr - z_hat_prev))),
    }


def run_experiment(config: ExperimentConfig) -> ExperimentLog:
    """Execute the full template-diffusion experiment."""
    log = ExperimentLog(config=asdict(config))
    config.out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = config.out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if config.use_wandb:
        try:
            wandb = importlib.import_module("wandb")
            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config={k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            )
            wandb_run.log({"event": "experiment_started", "pdb_id": config.pdb_id})
        except Exception as e:  # noqa: BLE001
            logger.warning("W&B init failed, continuing without W&B logging: %s", e)
            log.errors.append(f"W&B init failed: {e}")
            wandb_run = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config.cache, device)
    mol_dir = config.cache / "mols"

    logger.info("=== Round 1: MSA (no template) -> Diffusion ===")
    log.timestamps["round1_start"] = datetime.now().isoformat()

    round1_yaml = work_dir / "round1_input.yaml"
    create_round1_yaml(config.sequence, config.chain_id, round1_yaml)

    round1_out = work_dir / "round1_out_internal"
    success = True
    try:
        processed_round1 = process_single_input(
            input_path=round1_yaml,
            out_dir=round1_out,
            cache=config.cache,
            use_msa_server=config.use_msa_server,
            msa_server_url=config.msa_server_url,
        )
        batch_round1 = load_single_batch(processed_round1, mol_dir=mol_dir, device=device)
        out_round1 = run_forward_with_seed(
            model=model,
            feats=batch_round1,
            recycling_steps=config.recycling_steps,
            sampling_steps=config.sampling_steps,
            diffusion_samples=config.diffusion_samples,
            init_state=None,
            use_template_module=False,  # first iteration: MSA-only
        )
    except Exception as e:  # noqa: BLE001
        log.errors.append(f"Round 1 internal inference failed: {e}")
        success = False

    log.timestamps["round1_end"] = datetime.now().isoformat()

    if not success:
        log.errors.append("Round 1 prediction failed")
        if wandb_run is not None:
            wandb_run.log({"round": 1, "success": 0, "pdb_id": config.pdb_id})
            wandb_run.finish()
        return log

    round1_struct_path = work_dir / "round1_model_0.cif"
    write_prediction_structure(
        processed=processed_round1,
        batch=batch_round1,
        pred_coords=out_round1["sample_atom_coords"][0],
        out_cif=round1_struct_path,
    )
    np.savez_compressed(
        work_dir / "round1_embeddings.npz",
        s=out_round1["s"].detach().cpu().numpy(),
        z=out_round1["z"].detach().cpu().numpy(),
    )
    torch.save(out_round1["pairformer_trace"], work_dir / "round1_pairformer_trace.pt")
    log.round1 = {
        "structure_path": str(round1_struct_path),
        "success": True,
        "pairformer_trace_path": str(work_dir / "round1_pairformer_trace.pt"),
    }
    if wandb_run is not None:
        wandb_run.log(
            {
                "round": 1,
                "success": 1,
                "pdb_id": config.pdb_id,
                "structure_path": str(round1_struct_path),
            }
        )

    # Cache backbone atom mask once from the processed structure (atom
    # ordering is identical across iterations for the same sequence/chain).
    try:
        round1_structure = StructureV2.load(
            processed_round1.targets_dir / f"{batch_round1['record'][0].id}.npz"
        ).remove_invalid_chains()
        backbone_mask_np = backbone_atom_mask_from_structure(round1_structure)
    except Exception as e:
        log.errors.append(f"Round 1 backbone-mask error: {e}")
        backbone_mask_np = None

    # Per-sample backbone coords for the round-1 prediction ensemble. These
    # serve as the "previous structure ensemble" for iter 1 of round 2
    # (Protenix-style: always compare against the immediately preceding
    # iteration, not a fixed reference).
    prev_bb: torch.Tensor | None = None
    if backbone_mask_np is not None:
        try:
            prev_bb = ensemble_backbone_coords(
                out_round1["sample_atom_coords"],
                batch_round1["atom_pad_mask"][0],
                backbone_mask_np,
            )
        except Exception as e:
            log.errors.append(f"Round 1 backbone-extract error: {e}")
            prev_bb = None

    logger.info("Round 1 complete. Structure: %s", round1_struct_path)
    prev_embeddings = {
        "s": out_round1["s"].detach().cpu().numpy(),
        "z": out_round1["z"].detach().cpu().numpy(),
    }

    # Copy round1 structure as template for round2 iter 0
    template_path = work_dir / "template_0.cif"
    import shutil

    shutil.copy(round1_struct_path, template_path)

    # Locate the MSA CSV produced by the round-1 MMseqs2 call. Modes that ask
    # for "with MSA" in round 2 reuse this file rather than regenerating or
    # falling back to the 1-row dummy.
    round1_msa_dir = round1_out / "msa"
    candidate_csvs = sorted(round1_msa_dir.glob("*.csv"))
    round1_msa_csv: Path | None = candidate_csvs[0] if candidate_csvs else None
    if round1_msa_csv is None:
        log.errors.append(f"Round 1 MSA CSV not found under {round1_msa_dir}")

    logger.info("=== Round 2 mode: %s ===", config.mode)

    for m in range(1, config.M + 1):
        iter_start = datetime.now().isoformat()
        logger.info("--- Round 2 iteration %d/%d ---", m, config.M)

        round2_yaml = work_dir / f"round2_iter{m}_input.yaml"
        use_template_for_iter = config.mode in {
            "msa_then_template_no_msa",
            "msa_then_template_with_msa",
        }
        # Modes whose name ends in ``_with_msa`` or is ``msa_only_no_template``
        # feed the real round-1 MSA back into the MSA module in round 2.
        # All other modes use the Boltz dummy (1-row) MSA via ``msa: "empty"``.
        use_real_msa_for_iter = config.mode in {
            "msa_then_template_with_msa",
            "msa_only_no_template",
        }
        if use_real_msa_for_iter:
            if round1_msa_csv is None:
                raise RuntimeError(
                    f"Mode {config.mode} requires the round-1 MSA CSV but "
                    f"none was found under {round1_msa_dir}"
                )
            msa_source = str(round1_msa_csv.resolve())
        else:
            msa_source = "empty"
        create_round2_yaml(
            config.sequence,
            config.chain_id,
            template_path if use_template_for_iter else None,
            round2_yaml,
            msa_source=msa_source,
        )

        round2_out = work_dir / f"round2_iter{m}_out_internal"
        success = True
        try:
            processed_round2 = process_single_input(
                input_path=round2_yaml,
                out_dir=round2_out,
                cache=config.cache,
                use_msa_server=False,
                msa_server_url=config.msa_server_url,
            )
            batch_round2 = load_single_batch(processed_round2, mol_dir=mol_dir, device=device)
            recycle_steps_iter = (
                1 if config.mode == "msa_then_recycle5_no_template" else config.N
            )
            out_round2 = run_forward_with_seed(
                model=model,
                feats=batch_round2,
                recycling_steps=recycle_steps_iter,
                sampling_steps=config.sampling_steps,
                diffusion_samples=config.diffusion_samples,
                init_state=(
                    torch.from_numpy(prev_embeddings["s"]).to(device),
                    torch.from_numpy(prev_embeddings["z"]).to(device),
                ),
                use_template_module=use_template_for_iter,
            )
        except Exception as e:  # noqa: BLE001
            log.errors.append(f"Round 2 iter {m} internal inference failed: {e}")
            success = False
        iter_end = datetime.now().isoformat()

        iter_log = {
            "iteration": m,
            "mode": config.mode,
            "use_template": use_template_for_iter,
            "use_real_msa": use_real_msa_for_iter,
            "msa_source": msa_source,
            "recycling_steps": recycle_steps_iter,
            "template_path": str(template_path),
            "success": success,
            "start": iter_start,
            "end": iter_end,
        }

        if not success:
            log.errors.append(f"Round 2 iter {m} prediction failed")
            log.round2_iterations.append(iter_log)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "pdb_id": config.pdb_id,
                        "round2/iteration": m,
                        "round2/success": 0,
                    }
                )
            continue

        curr_struct_path = work_dir / f"round2_iter{m}_model_0.cif"
        write_prediction_structure(
            processed=processed_round2,
            batch=batch_round2,
            pred_coords=out_round2["sample_atom_coords"][0],
            out_cif=curr_struct_path,
        )
        np.savez_compressed(
            work_dir / f"round2_iter{m}_embeddings.npz",
            s=out_round2["s"].detach().cpu().numpy(),
            z=out_round2["z"].detach().cpu().numpy(),
        )
        trace_path = work_dir / f"round2_iter{m}_pairformer_trace.pt"
        torch.save(out_round2["pairformer_trace"], trace_path)
        iter_log["structure_path"] = str(curr_struct_path)
        iter_log["pairformer_trace_path"] = str(trace_path)

        # Per-sample backbone coords for the current iteration's ensemble.
        curr_bb: torch.Tensor | None = None
        if backbone_mask_np is not None:
            try:
                curr_bb = ensemble_backbone_coords(
                    out_round2["sample_atom_coords"],
                    batch_round2["atom_pad_mask"][0],
                    backbone_mask_np,
                )
            except Exception as e:
                iter_log["structure_extract_error"] = str(e)
                log.errors.append(f"Round 2 iter {m} backbone-extract error: {e}")

        # Structure diff: current ensemble vs previous iteration's ensemble
        # (Protenix-style min-over-references, then mean/std across current).
        if curr_bb is not None and prev_bb is not None:
            try:
                structure_diff, structure_diff_std = pairwise_structure_diff(
                    curr_bb, prev_bb
                )
                iter_log["structure_diff"] = structure_diff
                iter_log["structure_diff_std"] = structure_diff_std
                log.rmsds.append(structure_diff)
                logger.info(
                    "structure_diff: %.4f ± %.4f Å", structure_diff, structure_diff_std
                )
            except Exception as e:
                iter_log["structure_diff_error"] = str(e)
                log.errors.append(f"Round 2 iter {m} structure_diff error: {e}")

        curr_embeddings = {
            "s": out_round2["s"].detach().cpu().numpy(),
            "z": out_round2["z"].detach().cpu().numpy(),
        }
        if prev_embeddings is not None and curr_embeddings is not None:
            try:
                emb_change = compute_embedding_change(prev_embeddings, curr_embeddings)
                iter_log["s_diff"] = emb_change["s_diff"]
                iter_log["z_diff"] = emb_change["z_diff"]
                iter_log["s_norm_diff"] = emb_change["s_norm_diff"]
                iter_log["z_norm_diff"] = emb_change["z_norm_diff"]
                log.embedding_changes.append(
                    {
                        "iteration": m,
                        **{k: emb_change[k] for k in (
                            "s_diff", "z_diff", "s_norm_diff", "z_norm_diff"
                        )},
                    }
                )
            except Exception as e:
                iter_log["embedding_error"] = str(e)
                log.errors.append(f"Round 2 iter {m} embedding change error: {e}")
        elif curr_embeddings is None:
            iter_log["embedding_error"] = "Current embeddings not found"
            log.errors.append(f"Round 2 iter {m} embeddings not found")

        if wandb_run is not None:
            payload = {
                "pdb_id": config.pdb_id,
                "iteration": m,
                "success": 1,
            }
            for key in (
                "structure_diff",
                "structure_diff_std",
                "s_diff",
                "z_diff",
                "s_norm_diff",
                "z_norm_diff",
            ):
                if key in iter_log:
                    payload[key] = iter_log[key]
            wandb_run.log(payload)

        # Keep template fixed to the initial MSA-only output (template_0.cif).
        # Do not update template_path with each loop iteration.

        log.round2_iterations.append(iter_log)
        # Advance "previous" state for next iteration's comparison.
        if curr_embeddings is not None:
            prev_embeddings = curr_embeddings
        if curr_bb is not None:
            prev_bb = curr_bb

    log.timestamps["experiment_end"] = datetime.now().isoformat()
    if wandb_run is not None:
        wandb_run.log(
            {
                "pdb_id": config.pdb_id,
                "experiment/num_errors": len(log.errors),
                "experiment/num_iterations": len(log.round2_iterations),
            }
        )
        wandb_run.finish()
    return log


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Template-Diffusion relationship experiment"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Protein sequence (single chain)",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=10,
        help="Number of template-only iterations in round 2 (default: 5)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=3,
        help="Recycling steps (pairformer passes) per round 2 iteration (default: 3)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./template_diffusion_experiment"),
        help="Output directory",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("~/.boltz").expanduser(),
        help="Boltz cache directory",
    )
    parser.add_argument(
        "--no_msa_server",
        action="store_true",
        help="Disable MSA server (requires pre-computed MSA for round 1)",
    )
    parser.add_argument(
        "--msa_server_url",
        type=str,
        default="https://api.colabfold.com",
        help="MSA server URL",
    )
    parser.add_argument(
        "--recycling_steps",
        type=int,
        default=1,
        help="Recycling steps for round 1 (default: 3)",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=200,
        help="Diffusion sampling steps",
    )
    parser.add_argument(
        "--diffusion_samples",
        type=int,
        default=1,
        help="Number of diffusion samples per prediction",
    )
    parser.add_argument(
        "--pdb_id",
        type=str,
        default="1u0p",
        help="PDB identifier tag to log to W&B (default: 1u0p).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="msa_then_template_no_msa",
        choices=[
            "msa_then_template_no_msa",
            "msa_then_template_with_msa",
            "msa_then_recycle5_no_template",
            "msa_only_no_template",
        ],
        help=(
            "Experiment mode: "
            "msa_then_template_no_msa | "
            "msa_then_template_with_msa | "
            "msa_then_recycle5_no_template | "
            "msa_only_no_template"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42). Use -1 for no seed.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="boltz-template-diffusion",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team (optional).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (optional).",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        sequence=args.sequence,
        M=args.M,
        N=args.N,
        out_dir=args.out_dir,
        cache=args.cache,
        use_msa_server=not args.no_msa_server,
        msa_server_url=args.msa_server_url,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
        pdb_id=args.pdb_id,
        mode=args.mode,
        seed=args.seed if args.seed >= 0 else None,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

    log = run_experiment(config)

    # Save log (convert Path objects to strings for JSON)
    def _serialize(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        return obj

    log_path = config.out_dir / "experiment_log.json"
    with log_path.open("w") as f:
        json.dump(
            _serialize(
                {
                    "config": log.config,
                    "round1": log.round1,
                    "round2_iterations": log.round2_iterations,
                    "rmsds": log.rmsds,
                    "embedding_changes": log.embedding_changes,
                    "timestamps": log.timestamps,
                    "errors": log.errors,
                }
            ),
            f,
            indent=2,
        )
    logger.info("Experiment log saved to %s", log_path)

    if log.errors:
        logger.warning("Experiment completed with %d error(s)", len(log.errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

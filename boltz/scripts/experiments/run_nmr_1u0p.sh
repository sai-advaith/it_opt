#!/usr/bin/env bash
# Example: NMR-restraint embedding optimization for target 1u0p.
#
# Run from the boltz/ directory, in an environment where this fork is installed
# (see IT_Opt_Boltz.md). On a cluster you may first need `module load cuda conda`
# and `conda activate <env>`.
#
# WHAT TO CHANGE for a different target:
#   SEQUENCE        -> the chain sequence (metadata/<ID>/<ID>.json "seq")
#   PDB_ID          -> the target id
#   NMR_REFERENCE   -> nmr_pipeline_inputs/pdbs/<ID>/<ID>_fixed.pdb
#   NMR_RESTRAINTS  -> nmr_pipeline_inputs/restraints/<ID>/<ID>.csv
# Prepare a new target's files with:  python scripts/experiments/prepare_nmr_target.py <ID>
set -euo pipefail

SEQUENCE="GYIPEAPRDGQAYVRKDGEWVLLSTFL"
PDB_ID="1u0p"
NMR_REFERENCE="nmr_pipeline_inputs/pdbs/${PDB_ID}/${PDB_ID}_fixed.pdb"
NMR_RESTRAINTS="nmr_pipeline_inputs/restraints/${PDB_ID}/${PDB_ID}.csv"
OUT_DIR="out_nmr_${PDB_ID}"

python scripts/experiments/boltz_embedding_nmr_optimization.py \
    --sequence "${SEQUENCE}" \
    --pdb_id "${PDB_ID}" \
    --nmr_reference_pdb "${NMR_REFERENCE}" \
    --nmr_restraint_file "${NMR_RESTRAINTS}" \
    --out_dir "${OUT_DIR}" \
    --outer_steps 20 \
    --inner_steps 1 \
    --stop_opt_after_t 160 \
    --lr 0.025 \
    --max_grad_norm 0.6 \
    --batch_size 16 \
    --lambda_violation 0.05 \
    --lambda_bond_length 0.01 \
    --lambda_anchor_s 1e-3 \
    --lambda_anchor_z 1e-3 \
    --no_wandb
    # NMR assets (fragments/names pickles) resolve automatically to the copy
    # vendored in the package; override with --nmr_assets_dir if needed.
    # To log to Weights & Biases, drop --no_wandb and add:
    #   --wandb_project <proj> --wandb_entity <entity>   (run `wandb login` first)

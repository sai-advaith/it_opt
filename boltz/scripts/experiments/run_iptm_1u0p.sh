#!/usr/bin/env bash
# Example: iPTM/pTM embedding optimization for target 1u0p.
#
# Run from the boltz/ directory, in an environment where this fork is installed
# (see IT_Opt_Boltz.md). On a cluster you may first need `module load cuda conda`
# and `conda activate <env>`.
#
# iPTM optimization needs NO uploaded files — just a sequence. The objective is
# the model's own confidence (0.8*iPTM + 0.2*pTM by default).
#
# WHAT TO CHANGE:
#   SEQUENCE  -> your chain sequence. (pTM is the meaningful signal for a single
#                chain; iPTM only matters for complexes.)
#   PDB_ID    -> a label for outputs (any string).
#
# For a MULTI-CHAIN complex (where iPTM is informative), repeat --sequence and
# --chain_id, e.g.:
#     --sequence "CHAIN_A_SEQ" --sequence "CHAIN_B_SEQ" \
#     --chain_id A --chain_id B \
set -euo pipefail

SEQUENCE="GYIPEAPRDGQAYVRKDGEWVLLSTFL"
PDB_ID="1u0p"
OUT_DIR="out_iptm_${PDB_ID}"

python scripts/experiments/boltz_embedding_iptm_optimization.py \
    --sequence "${SEQUENCE}" \
    --pdb_id "${PDB_ID}" \
    --out_dir "${OUT_DIR}" \
    --outer_steps 5 \
    --inner_steps 1 \
    --stop_opt_after_t 160 \
    --lr 1e-3 \
    --max_grad_norm 1.0 \
    --batch_size 16 \
    --w_iptm 0.8 \
    --w_ptm 0.2 \
    --no_wandb
    # To also optimize the single representation s, add --optimize_s.
    # To log to Weights & Biases, drop --no_wandb and add:
    #   --wandb_project <proj> --wandb_entity <entity>   (run `wandb login` first)

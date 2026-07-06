#!/bin/bash
set -euo pipefail

# Repo (boltz/) root, derived from this script's location (scripts/experiments/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="${ROOT}/scripts/experiments/run_1u0p_modes.sbatch"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
SUBMIT_LOG="${LOG_DIR}/submit_1u0p_modes_${TS}.log"

# Log everything from this launcher invocation (stdout + stderr)
exec > >(tee -a "${SUBMIT_LOG}") 2>&1

echo "Submitting sequence/mode matrix: ${SBATCH_SCRIPT}"
SUBMIT_OUTPUT="$(sbatch "${SBATCH_SCRIPT}")"
echo "${SUBMIT_OUTPUT}"
echo "Launcher log saved to: ${SUBMIT_LOG}"

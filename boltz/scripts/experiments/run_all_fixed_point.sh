#!/bin/bash
# Run the full fixed-point sweep (13 PDBs x 4 modes = 52 runs) serially on the
# local GPU, logging each run to the boltz_fixed_point wandb project.
#
# Layout under logs/fixed_point_sweep/:
#   driver.out                  - stdout/stderr of this script itself
#   sweep_summary.log           - human-readable one-line-per-run timeline
#   sweep_manifest.csv          - machine-readable status table (updated live)
#   <run_name>.log              - full stdout/stderr of that python run
#   <run_name>.status           - one of: RUNNING, SUCCESS, FAIL
#   <run_name>.meta             - key=value metadata (rc, duration, wandb_url, ...)
#   <run_name>/                 - experiment output dir (structures, embeddings, json)
#
# Failure handling:
#   - python failures are captured (set +e around the call); the sweep keeps going
#   - each failure writes a .status=FAIL + first error line into .meta
#   - a run is considered "done" (for idempotent skip) only if
#     experiment_log.json exists AND .status=SUCCESS
#   - final post-mortem prints counts and failed-run error snippets

set -uo pipefail

# Repo (boltz/) root, derived from this script's location (scripts/experiments/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="${ROOT}/scripts/experiments/template_diffusion_experiment.py"
SWEEP_ROOT="${ROOT}/logs/fixed_point_sweep"
SUMMARY_LOG="${SWEEP_ROOT}/sweep_summary.log"
MANIFEST_CSV="${SWEEP_ROOT}/sweep_manifest.csv"

mkdir -p "${SWEEP_ROOT}"

# Initialize manifest with header if new.
if [ ! -f "${MANIFEST_CSV}" ]; then
  echo "timestamp,pdb_id,mode,run_name,status,rc,duration_s,wandb_url,log_path,error_snippet" > "${MANIFEST_CSV}"
fi

# Activate the boltz conda env (matches sbatch behavior).
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda
module load conda
conda activate boltz

# Wandb credentials: export WANDB_API_KEY in your shell before running,
# e.g.  export WANDB_API_KEY="<your-key>"   (or run `wandb login`).
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment, or run with WANDB_MODE=disabled}"

PROJECT="boltz_fixed_point"
M=10
N=1
SAMPLING_STEPS=200
DIFFUSION_SAMPLES=1

declare -a PDB_IDS=(
  "1d3z"
  "1dec"
  "1u0p"
  "2kif"
  "6sow"
  "2li3"
  "1rkl"
  "1pqx"
  "2lf2"
  "2l82"
  "1u7r"
  "6s7n"
  "6cyh"
)

declare -a SEQS=(
  "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
  "APRLPQCQGDDQEKCLCNKDECPPGQCRFPRGDADPYCE"
  "GYIPEAPRDGQAYVRKDGEWVLLSTFL"
  "MDQFLVQIFAVIHQIPKGKVSTYGEIAKMAGYPGYARHVGKALGNLPEGSKLPWFRVINSQGKISLKGRDLDRQKQKLEAEGIEVSEIGKIALRKYKWQPLE"
  "SDNKFNKEQQNAFYEILHLPNLTEEQRNGFIQSLKDDPSVSKEILAEAKKLNDAQAPK"
  "GSGCMPEYCAGQCRGKVSQDYCLKNCRCIR"
  "MISDEQLNSLAITFGIVMMTLIVIYHAVDSTMSPKN"
  "MKIISISETPNHNTMKITLSESREGMTSDTYTKVDDSQPAFINDILKVEGVKSIFHVMDFISVDKENDANWETVLPKVEAVFELEHHHHHH"
  "MRTDLALDFSVNKENKTITIKREFAAVRAIVWEAFTRAEILDQWWAPKPWKAKTKSMDFKEGGTWLYAMVGPNGEEHWSICEYAIIKPIERFTGKDGFTDASGKLNTEMPRSNWDMRFIDKGEITEVQYHISYDDVAQLEATIQMGFKEGITMAMENLDELLVSGKKLEHHHHHH"
  "MSQIFVVFSSDPEILKEIVREIKRQGVRVVLLYSDQDEKRRRERLEEFEKQGVDVRTVEDKEDFRENIREIWERYPQLDVVVIVTTDDKEWIKDFIEEAKERGVEVFVVYNNKDDDRRKEAQQEFRSDGVDVRTVSDKEELIEQVRRFVRKVGSLEHHHHHH"
  "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG"
  "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
  "EVETFAFQAEIAQLMSLIINTFYSNKEIFLRELISNSSDALDKIRYESLTDPSKLDSGKELHINLIPNKQDRTLTIVDTGIGMTKADLINNLGTIAKSGTKAFMEALQAGADISMIGQFGVGFYSAYLVAEKVTVITKHNDDEQYAWESSAGGSFTVRTDTGEPMGRGTKVILHLKEDQTEYLEERRIKEIVKKHSQFIGYPITLFVEKE"
)

declare -a MODES=(
  "msa_then_template_no_msa"
  "msa_then_template_with_msa"
  "msa_then_recycle5_no_template"
  "msa_only_no_template"
)

csv_escape() {
  # Escape a field for CSV (doubles embedded quotes, wraps in quotes).
  local v="${1//\"/\"\"}"
  printf '"%s"' "${v}"
}

extract_wandb_url() {
  local log_file="$1"
  # Line format printed by wandb SDK: "🚀 View run <name> at: <url>"
  grep -oE 'https://wandb\.ai/[^ ]+/runs/[A-Za-z0-9]+' "${log_file}" 2>/dev/null | head -n1
}

extract_error_snippet() {
  local log_file="$1"
  # Find the first "Error" / "Exception" / traceback line as a one-liner.
  grep -iE 'error|traceback|exception|cuda out of memory|failed' "${log_file}" 2>/dev/null \
    | grep -vi 'wandb: ERROR' \
    | head -n1 \
    | tr '\n' ' ' \
    | cut -c1-300
}

total=$(( ${#PDB_IDS[@]} * ${#MODES[@]} ))
idx_run=0
n_success=0
n_fail=0
n_skip=0
ts_start=$(date +%s)
echo "=== sweep started at $(date -Iseconds) on $(hostname) ===" | tee -a "${SUMMARY_LOG}"
echo "Total runs planned: ${total}" | tee -a "${SUMMARY_LOG}"
echo "Manifest: ${MANIFEST_CSV}" | tee -a "${SUMMARY_LOG}"

for idx in "${!PDB_IDS[@]}"; do
  pdb_id="${PDB_IDS[$idx]}"
  seq="${SEQS[$idx]}"

  for mode in "${MODES[@]}"; do
    idx_run=$(( idx_run + 1 ))
    run_name="${pdb_id}_${mode}"
    out_dir="${SWEEP_ROOT}/${run_name}"
    run_log="${SWEEP_ROOT}/${run_name}.log"
    status_file="${SWEEP_ROOT}/${run_name}.status"
    meta_file="${SWEEP_ROOT}/${run_name}.meta"

    # Idempotent skip: only consider a run done if it produced the final
    # experiment_log.json AND previously flagged SUCCESS.
    if [ -f "${out_dir}/experiment_log.json" ] \
       && [ -f "${status_file}" ] \
       && grep -q '^SUCCESS$' "${status_file}"; then
      n_skip=$(( n_skip + 1 ))
      echo "[$(date -Iseconds)] [${idx_run}/${total}] SKIP already done: ${run_name}" | tee -a "${SUMMARY_LOG}"
      continue
    fi

    # Mark as RUNNING (useful if you inspect while it's in flight).
    echo "RUNNING" > "${status_file}"
    t0=$(date +%s)
    printf 'pdb_id=%s\nmode=%s\nrun_name=%s\nstart_iso=%s\nstatus=RUNNING\n' \
      "${pdb_id}" "${mode}" "${run_name}" "$(date -Iseconds)" > "${meta_file}"
    echo "[$(date -Iseconds)] [${idx_run}/${total}] START ${run_name}" | tee -a "${SUMMARY_LOG}"

    set +e
    python "${SCRIPT}" \
      --sequence "${seq}" \
      --pdb_id "${pdb_id}" \
      --mode "${mode}" \
      --M "${M}" \
      --N "${N}" \
      --sampling_steps "${SAMPLING_STEPS}" \
      --diffusion_samples "${DIFFUSION_SAMPLES}" \
      --wandb_project "${PROJECT}" \
      --wandb_run_name "${run_name}" \
      --out_dir "${out_dir}" \
      > "${run_log}" 2>&1
    rc=$?
    set -e

    t1=$(date +%s)
    dur=$(( t1 - t0 ))
    wandb_url="$(extract_wandb_url "${run_log}")"
    error_snippet=""

    # A run is SUCCESS iff python exited 0 AND experiment_log.json was written.
    if [ "${rc}" -eq 0 ] && [ -f "${out_dir}/experiment_log.json" ]; then
      status="SUCCESS"
      n_success=$(( n_success + 1 ))
      echo "SUCCESS" > "${status_file}"
    else
      status="FAIL"
      n_fail=$(( n_fail + 1 ))
      echo "FAIL" > "${status_file}"
      error_snippet="$(extract_error_snippet "${run_log}")"
    fi

    # Persist per-run metadata.
    {
      printf 'pdb_id=%s\n' "${pdb_id}"
      printf 'mode=%s\n' "${mode}"
      printf 'run_name=%s\n' "${run_name}"
      printf 'start_iso=%s\n' "$(date -Iseconds -d "@${t0}")"
      printf 'end_iso=%s\n' "$(date -Iseconds -d "@${t1}")"
      printf 'duration_s=%s\n' "${dur}"
      printf 'rc=%s\n' "${rc}"
      printf 'status=%s\n' "${status}"
      printf 'wandb_url=%s\n' "${wandb_url}"
      printf 'log_path=%s\n' "${run_log}"
      printf 'out_dir=%s\n' "${out_dir}"
      if [ -n "${error_snippet}" ]; then
        printf 'error_snippet=%s\n' "${error_snippet}"
      fi
    } > "${meta_file}"

    # Append to CSV manifest (csv-escaped).
    {
      ts="$(date -Iseconds)"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${ts}" \
        "${pdb_id}" \
        "${mode}" \
        "${run_name}" \
        "${status}" \
        "${rc}" \
        "${dur}" \
        "$(csv_escape "${wandb_url}")" \
        "$(csv_escape "${run_log}")" \
        "$(csv_escape "${error_snippet}")"
    } >> "${MANIFEST_CSV}"

    if [ "${status}" = "SUCCESS" ]; then
      echo "[$(date -Iseconds)] [${idx_run}/${total}] DONE  ${run_name} in ${dur}s  ${wandb_url}" | tee -a "${SUMMARY_LOG}"
    else
      echo "[$(date -Iseconds)] [${idx_run}/${total}] FAIL  ${run_name} in ${dur}s (rc=${rc}); see ${run_log}" | tee -a "${SUMMARY_LOG}"
      if [ -n "${error_snippet}" ]; then
        echo "           error: ${error_snippet}" | tee -a "${SUMMARY_LOG}"
      fi
    fi
  done
done

ts_end=$(date +%s)
total_dur=$(( ts_end - ts_start ))

echo "" | tee -a "${SUMMARY_LOG}"
echo "=== sweep finished at $(date -Iseconds); total duration ${total_dur}s ===" | tee -a "${SUMMARY_LOG}"
echo "Totals: success=${n_success} fail=${n_fail} skip=${n_skip} planned=${total}" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"

# Post-mortem: list failed runs with their first error snippet.
if [ "${n_fail}" -gt 0 ]; then
  echo "--- FAILED RUNS ---" | tee -a "${SUMMARY_LOG}"
  for meta in "${SWEEP_ROOT}"/*.meta; do
    if grep -q '^status=FAIL$' "${meta}" 2>/dev/null; then
      name="$(grep -E '^run_name=' "${meta}" | cut -d= -f2-)"
      err="$(grep -E '^error_snippet=' "${meta}" | cut -d= -f2-)"
      log="$(grep -E '^log_path=' "${meta}" | cut -d= -f2-)"
      echo "  - ${name}" | tee -a "${SUMMARY_LOG}"
      [ -n "${err}" ] && echo "      error: ${err}" | tee -a "${SUMMARY_LOG}"
      echo "      log:   ${log}" | tee -a "${SUMMARY_LOG}"
    fi
  done
fi

echo "" | tee -a "${SUMMARY_LOG}"
echo "Manifest CSV: ${MANIFEST_CSV}" | tee -a "${SUMMARY_LOG}"
echo "Per-run logs: ${SWEEP_ROOT}/<run_name>.log" | tee -a "${SUMMARY_LOG}"

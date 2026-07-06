# Boltz Experiments

Scripts in this directory all build on the Boltz-2 model checkpoint
(`~/.boltz/boltz2_conf.ckpt`) and reuse its internals (`input_embedder`,
`rel_pos`, `msa_module`, `pairformer_module`, `structure_module`,
`confidence_module`, etc.) instead of going through the `boltz predict` CLI.
They fall into three groups:

1. **Template–diffusion fixed-point experiment** — one Python driver plus
   several launcher shells that sweep a PDB × mode matrix.
2. **Embedding-space optimization** — two standalone Python scripts that
   optimize the pairformer trunk outputs `(s, z)` directly against an
   objective evaluated on the diffusion output.

Activate the environment that has this fork installed first (see
[`../../IT_Opt_Boltz.md`](../../IT_Opt_Boltz.md) for setup). On an HPC cluster
you may first need to load the relevant modules, e.g.:

```bash
module load cuda conda     # cluster-specific; skip if not applicable
conda activate boltz
```

---

## 1. `template_diffusion_experiment.py`

Driver for the **template ↔ diffusion fixed-point experiment**. It asks:
if we feed a model-predicted structure back in as a template (with the MSA
turned off), does the next prediction converge, drift, or oscillate?

### Pipeline

* **Round 1** — MSA from the ColabFold/mmseqs2 server, **no template**.
  Produces `structure_0` and caches its `(s, z)` trunk output as
  `round1_embeddings.npz`.
* **Round 2** — `M` iterations. Each iteration rebuilds the YAML, re-runs
  `process_inputs`, re-featurizes, then runs the trunk + diffusion. The
  trunk is **warm-started** from the previous iteration's `(s, z)` (i.e.
  `init_state=(s_prev, z_prev)` inside `run_forward_with_seed`), and the
  template used every iteration is `template_0.cif` (the round-1 output —
  the template is intentionally **not** updated).
* Per iteration it records:
  * Backbone (N, CA, C, O) RMSD between the current diffusion-sample
    ensemble and the previous iteration's ensemble, computed
    Protenix-style: pairwise `weighted_rigid_align` RMSDs, min over the
    reference axis, mean/std over current samples
    (`pairwise_structure_diff`).
  * Embedding drift metrics `s_diff`, `z_diff`, `s_norm_diff`, `z_norm_diff`
    (`compute_embedding_change`) matching Protenix's
    `fixed_point_analysis.py`.
  * Full pairformer trace per recycle iteration.
* Logs everything to `experiment_log.json`, per-iteration CIFs under
  `work/`, and W&B (`boltz-template-diffusion` project by default).

### Modes

Passed via `--mode`; they differ only in what round 2 sees:

| Mode | Round-2 template | Round-2 MSA |
| --- | --- | --- |
| `msa_then_template_no_msa` (default) | round-1 output CIF | `"empty"` (dummy 1-row MSA) |
| `msa_then_template_with_msa` | round-1 output CIF | round-1 MSA CSV reused |
| `msa_then_recycle5_no_template` | none | `"empty"`, round 2 uses `recycling_steps=1` |
| `msa_only_no_template` | none | round-1 MSA CSV reused |

The MSA module is always run (matching training-time single-sequence
regime); modes that say "no MSA" just use the dummy 1-row MSA the
featurizer injects for `msa: "empty"`.

### Usage

```bash
python scripts/experiments/template_diffusion_experiment.py \
    --sequence "MVTPEGNV...LYFQ" \
    --pdb_id 1u0p \
    --mode msa_then_template_no_msa \
    --M 10 --N 1 \
    --sampling_steps 200 --diffusion_samples 1 \
    --out_dir ./out_1u0p
```

Main options: `--M` (round-2 iterations), `--N` (recycling per iteration),
`--sampling_steps`, `--diffusion_samples`, `--recycling_steps` (round 1),
`--no_msa_server`, `--msa_server_url`, `--no_wandb`,
`--wandb_project/--wandb_run_name`, `--seed`.

### Output layout

```
<out_dir>/
  experiment_log.json                 # full config + per-iter metrics + errors
  work/
    round1_input.yaml
    round1_out_internal/              # Boltz processed inputs for round 1
    round1_model_0.cif
    round1_embeddings.npz             # (s, z) trunk output
    round1_pairformer_trace.pt
    template_0.cif                    # copied from round1 output, reused every iter
    round2_iter{m}_input.yaml
    round2_iter{m}_out_internal/
    round2_iter{m}_model_0.cif
    round2_iter{m}_embeddings.npz
    round2_iter{m}_pairformer_trace.pt
```

---

## 2. `run_1u0p_modes.sbatch` + `run_1u0p_modes.sh`

Pair of shell scripts that sweeps `template_diffusion_experiment.py`
across **13 PDB IDs × 4 modes = 52 runs** on SLURM.

* `run_1u0p_modes.sbatch` — the SLURM job. Requests a GPU partition
  (`gpu100` in the script — change it to match your cluster), 1 GPU, 8 CPUs,
  48 GB RAM, 24 h walltime. Activates the conda env and loops PDBs × modes
  **sequentially in one job**. Uses `M=10`, `N=1`, `--sampling_steps 200`,
  `--diffusion_samples 1`, logging to W&B project
  `boltz_ref_template_pairformer_1`. Per-run output directory:
  `${ROOT}/template_diffusion_${pdb_id}_${mode}/` (set `ROOT` at the top of the
  script).
* `run_1u0p_modes.sh` — thin launcher: `sbatch`s the above and tees its
  own stdout/stderr to `logs/submit_1u0p_modes_<timestamp>.log`. Despite
  the file name, the matrix inside is not restricted to 1u0p — it covers
  13 PDBs (1d3z, 1dec, 1u0p, 2kif, 6sow, 2li3, 1rkl, 1pqx, 2lf2, 2l82,
  1u7r, 6s7n, 6cyh).

### Usage

```bash
bash scripts/experiments/run_1u0p_modes.sh
```

---

## 3. `run_all_fixed_point.sh`

Same PDB × mode matrix as the sbatch above, but **runs serially on the
local GPU** (no SLURM) with much richer per-run bookkeeping. Intended
for an interactive node or a pre-allocated GPU session.

Differences vs the sbatch variant:

* W&B project `boltz_fixed_point`. The script sets `WANDB_API_KEY` inline —
  replace it with your own key, or delete that line and export the key from
  your environment instead.
* Rich logging under `logs/fixed_point_sweep/`:
  * `sweep_summary.log` — human timeline (START / DONE / FAIL per run).
  * `sweep_manifest.csv` — machine-readable status table, appended live,
    columns `timestamp,pdb_id,mode,run_name,status,rc,duration_s,wandb_url,log_path,error_snippet`.
  * `<run_name>.log` — full stdout/stderr of that Python invocation.
  * `<run_name>.status` — `RUNNING` | `SUCCESS` | `FAIL`.
  * `<run_name>.meta` — key=value metadata (rc, duration, wandb_url, etc).
  * `<run_name>/` — the Python script's own `out_dir`.
* **Idempotent skip**: re-running skips any run with both
  `experiment_log.json` present *and* `.status=SUCCESS`, so you can
  Ctrl-C and resume.
* **Fail-soft**: one run's failure doesn't abort the sweep. A Python
  failure captures `rc`, extracts the first error/traceback line via
  `grep`, and keeps going.
* **Post-mortem**: at the end, lists failed runs with their error
  snippet and log path.

### Usage

```bash
bash scripts/experiments/run_all_fixed_point.sh
```

---

## 4. `boltz_embedding_nmr_optimization.py`

**Guided-sampling / post-hoc optimization** of Boltz-2 against an NMR
restraint loss. It treats the pairformer trunk outputs `(s, z)` as the
optimization variables and walks them through the reverse diffusion
trajectory.

It uses the same 3-level optimization structure as the other IT-opt drivers in
this repo (outer noise realisations → diffusion timesteps → inner embedding
updates):

```
outer loop   = fresh diffusion noise realisations (fresh x_T)
middle loop  = each diffusion timestep t
inner loop   = optimise (s_opt, z_opt) at that noise level
               - denoise atom_coords_noisy at t_hat (differentiable)
               - compute loss on x_0_hat
               - backprop, clip, Adam step on (s_opt, z_opt)
after inner  = take the actual Euler reverse step forward
```

`(s_opt, z_opt)` are initialized to a pairformer baseline `(s_base,
z_base)` computed once (zero-seed recycle). Optimisation is **frozen**
(inner loop skipped) for diffusion steps `>= stop_opt_after_t` to limit
distributional shift near the end of denoising.

### Loss stack (weighted sum)

| Term | Module | What it encodes |
| --- | --- | --- |
| `main_loss` | `NMRLossFunction` | NOE + optional methyl/amide RDC and relaxation restraints |
| `bond_length_loss` | `BondLengthLossFunction` | covalent geometry + clash penalty |
| `violation_loss` | `ViolationLossFunction` | OpenFold-style structural violations |
| `anchor_loss` | `AnchorLossFunction` | L2 regularization of `(s_opt, z_opt)` toward the pairformer baseline |

Each term's gradient norm w.r.t. both `s_opt` and `z_opt` is logged
separately each inner step, which makes this script useful as a
diagnostic for which objective is actually moving which embedding.

### Outputs

Per outer loop `k`:

* `optimized_outer{k}.cif` — final denoised structure for sample 0 of the
  batch.
* `embeddings_outer{k}.pt` — dict of `s_base, z_base, s_opt, z_opt` on CPU.
* `coords_outer{k}.pt` — full batch of final atom coords.
* Periodic `snapshot_outer{k}_step{step}.cif` snapshots along the
  diffusion trajectory.

Also writes the running loss history to `losses_nmr_boltz_checkpoint.pt`
(in the CWD, not `out_dir`) and logs every inner step to W&B
(`boltz-embedding-nmr-optim` by default).

### Usage

Inputs live under `nmr_pipeline_inputs/` (reference `<ID>_fixed.pdb` +
parsed NOE `<ID>.csv`); see the "Running NMR-restraint optimization" section of
[`../../IT_Opt_Boltz.md`](../../IT_Opt_Boltz.md) for how to obtain them
(`prepare_nmr_target.py`, or copy from the Protenix fork). The restraint file is
a **CSV** (read with `pd.read_csv`), not a `.tbl`. A ready example is
[`run_nmr_1u0p.sh`](run_nmr_1u0p.sh). Raw command:

```bash
python scripts/experiments/boltz_embedding_nmr_optimization.py \
    --sequence "GYIPEAPRDGQAYVRKDGEWVLLSTFL" \
    --pdb_id 1u0p \
    --nmr_reference_pdb  nmr_pipeline_inputs/pdbs/1u0p/1u0p_fixed.pdb \
    --nmr_restraint_file nmr_pipeline_inputs/restraints/1u0p/1u0p.csv \
    --out_dir ./out_nmr_1u0p \
    --outer_steps 20 --inner_steps 1 --stop_opt_after_t 160 \
    --lr 0.025 --max_grad_norm 0.6 --batch_size 16 \
    --lambda_violation 0.05 --lambda_bond_length 0.01 \
    --lambda_anchor_s 1e-3 --lambda_anchor_z 1e-3 \
    --no_wandb
```

Needs `boltz.experiments.nmr.assets` (fragment/name pickles) in either
`--nmr_assets_dir` or the package default, a reference PDB (hydrogens
and OXT atoms are stripped before loss computation so topology matches
Boltz heavy-atom output), and optionally `--atom_index_map_npy` to
permute Boltz atom order into the NMR loss's expected order.

---

## 5. `boltz_embedding_iptm_optimization.py`

Same 3-level optimisation skeleton as the NMR script, but the objective
is the model's own confidence metrics:

```
objective = 0.8 * iptm + 0.2 * ptm    # default weights; --w_iptm / --w_ptm configurable
loss      = -objective.mean()         # we maximise
```

At each inner step it

1. Denoises `atom_coords_noisy` to `x_0_hat` using the current
   `(s_opt, z_opt)`.
2. Runs `distogram_module(z_trunk)` to get distogram logits.
3. Runs `confidence_module` to get `ptm, iptm`.
4. Backprops `-objective` into `(s_opt, z_opt)` and Adam-steps.

By default only `z_opt` is optimized (kernels can leave `s_opt` without
a gradient path); pass `--optimize_s` to also update `s_opt`.

Accepts one **or more** `--sequence` flags so you can optimize the
iPTM of a multi-chain complex:

```bash
python scripts/experiments/boltz_embedding_iptm_optimization.py \
    --sequence "CHAIN_A_SEQ" --sequence "CHAIN_B_SEQ" \
    --chain_id A --chain_id B \
    --pdb_id mycomplex \
    --out_dir ./out_iptm_mycomplex \
    --outer_steps 5 --inner_steps 1 --stop_opt_after_t 160 \
    --lr 1e-3 --batch_size 16 \
    --w_iptm 0.8 --w_ptm 0.2
```

Outputs per outer loop `k`: `optimized_outer{k}.cif` and
`embeddings_outer{k}.pt`, plus running `losses_iptm_boltz_checkpoint.pt`
and W&B logs (`boltz-embedding-iptm-optim`). Errors out at startup if
the loaded checkpoint is missing `confidence_module` or
`distogram_module`.

---

## RMSD / embedding metric conventions (shared)

Both the fixed-point driver and the optimisation scripts reuse
`boltz.model.loss.diffusionv2.weighted_rigid_align` for optimal
superposition before RMSD. The fixed-point driver additionally uses the
Protenix reductions (min over reference samples, mean/std over current
samples; Frobenius-normalized `s/z_diff`), so numbers should be
directly comparable to Protenix's `fixed_point_analysis.py`.

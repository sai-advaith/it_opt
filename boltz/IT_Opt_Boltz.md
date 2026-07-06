# Inference-Time Optimization for Boltz-2

This fork extends Boltz-2 with **embedding-space inference-time optimization**:
treat the pairformer trunk outputs `(s, z)` as optimization variables and walk
them through the reverse diffusion trajectory under a differentiable objective
evaluated on the denoised structure.

Two objectives are wired up out of the box:

| Script | Objective | Use case |
| --- | --- | --- |
| `boltz_embedding_iptm_optimization.py` | maximize `0.8·iPTM + 0.2·pTM` | improve model self-confidence; complex docking |
| `boltz_embedding_nmr_optimization.py`  | minimize NMR restraint + geometry losses | bias sampling toward NMR-consistent structures |

The upstream Boltz potentials-based guidance hook
(`src/boltz/model/modules/diffusionv2.py:454–484`) is still intact for
physical / contact guidance during sampling.

For per-script reference (every CLI flag, the 3-level loop, output layout,
metric conventions), see [`scripts/experiments/README.md`](scripts/experiments/README.md).
This file is the quickstart.

---

## 1. Setup

Install **this fork** editable into a fresh environment so that `import boltz`
resolves to the code in this tree (not a separately installed copy):

```bash
# from the boltz/ directory of this repo
pip install -e .[cuda]        # drop [cuda] on CPU-only / non-CUDA hardware
```

On an HPC cluster you may first need to load the relevant modules, e.g.
`module load cuda conda` and activate your environment. Confirm `boltz`
resolves to this fork:

```bash
python -c "import boltz, os; print(os.path.abspath(boltz.__file__))"
# should print .../it_opt/boltz/src/boltz/__init__.py
```

### Model weights

This code runs on top of [Boltz-2](https://github.com/jwohlwend/boltz). The
weights are **publicly downloadable** (MIT-licensed). The scripts resolve the
cache directory in this order: the `--cache <dir>` flag → the `$BOLTZ_CACHE`
environment variable → `~/.boltz` (default), and load
`<cache>/boltz2_conf.ckpt`.

**Easiest — let Boltz fetch them on first run.** Any `boltz predict` call
downloads the weights into the cache directory automatically:

```bash
boltz predict examples/prot.yaml --use_msa_server   # populates ~/.boltz on first run
```

**Manual download** (e.g. on a node without internet at runtime), into the
cache directory:

```bash
mkdir -p ~/.boltz
# Structure/confidence checkpoint — the only one the IT-opt scripts need
wget -P ~/.boltz https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt
# CCD molecule cache used during featurization
wget -P ~/.boltz https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar
tar -xf ~/.boltz/mols.tar -C ~/.boltz        # extracts ~/.boltz/mols/
wget -P ~/.boltz https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl
```

**Where to place them:** everything goes in the cache directory the scripts
read from — by default:

```
~/.boltz/
├── boltz2_conf.ckpt     # <-- loaded by the IT-opt scripts (boltz2_aff.ckpt is NOT needed)
├── mols/                # extracted CCD molecule cache
├── mols.tar
└── ccd.pkl
```

To use a different location, set `BOLTZ_CACHE=/path/to/cache` or pass
`--cache /path/to/cache` to the scripts (the directory must contain
`boltz2_conf.ckpt`).

### NMR input-prep dependencies (optional)

Only needed to run `scripts/experiments/prepare_nmr_target.py` (scraping and
parsing NMR restraints). The core iPTM / NMR **optimization** scripts don't need
these — they only read the already-prepared `.csv` / `.pdb`.

The bioinformatics packages the prep script relies on — `gemmi`, `biopython`,
`rdkit`, `requests`, `pandas` — are already in this fork's `pyproject.toml`. It
needs just two more:

```bash
pip install pynmrstar==3.3.5
# PDBFixer (+ OpenMM). Via conda:  conda install -c conda-forge pdbfixer openmm
# or from source:
git clone https://github.com/openmm/pdbfixer.git
cd pdbfixer && python setup.py install && cd -
```

---

## 2. The optimization loop

Both scripts share the same 3-level skeleton:

```
outer loop   = fresh diffusion noise realisations (fresh x_T)
middle loop  = each diffusion timestep t
inner loop   = optimise (s_opt, z_opt) at that noise level
               - denoise atom_coords_noisy at t_hat (differentiable)
               - compute loss on x_0_hat
               - backprop, clip, Adam step on (s_opt, z_opt)
after inner  = take the actual Euler reverse step forward
```

`(s_opt, z_opt)` start from a pairformer baseline `(s_base, z_base)` (one
zero-seed recycle). Optimization is **frozen** for diffusion steps `t >=
stop_opt_after_t` to limit distribution shift near the end of denoising.

**Where the nested loop lives** (both drivers implement it inside `main()`):

| Loop | iPTM driver (`boltz_embedding_iptm_optimization.py`) | NMR driver (`boltz_embedding_nmr_optimization.py`) |
|---|---|---|
| Outer — fresh noise realisations | `:514` `for outer in range(cfg.outer_steps):` | `:624` `for outer in range(cfg.outer_steps):` |
| Middle — diffusion timesteps | `:519` `for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):` | `:635` (same) |
| Inner — Adam on `(s_opt, z_opt)` | `:538` `for inner in range(cfg.inner_steps if do_opt else 0):` | `:654` (same) |

**Where the optimization actually happens** (inside the inner loop):

| Step | iPTM driver | NMR driver |
|---|---|---|
| Adam optimizer over the embeddings | `:536` `torch.optim.Adam(params, lr=cfg.lr)` | `:649` `torch.optim.Adam([s_opt, z_opt], lr=cfg.lr)` |
| Differentiable forward (denoise → `x_0_hat`) | `:544` `denoise_at_timestep(...)` | `:660` `denoise_at_timestep(...)` |
| Scalar loss | `:576` `loss = -objective.mean()` (objective = `w_iptm·iPTM + w_ptm·pTM` from the distogram/confidence modules at `:557`/`:561`) | `:679` `main_loss` → `:700` `total_loss = main + bond + viol + anchor` |
| Backprop → clip → step | `:589` `loss.backward()` → `:590–592` clip → `:593` `optimizer.step()` | `:701` `total_loss.backward()` → `:703–704` clip → `:705` `optimizer.step()` |
| Euler reverse step (after inner loop) | after the inner loop | `:777` |

(Line numbers are in `scripts/experiments/`.)

---

## 3. Running iPTM / pTM optimization

**Ready-to-run example:** [`scripts/experiments/run_iptm_1u0p.sh`](scripts/experiments/run_iptm_1u0p.sh)
— just `bash` it, then edit `SEQUENCE` / `PDB_ID` at the top for your own target.
**iPTM optimization needs no uploaded files** — only a sequence (pTM is the
meaningful signal for a single chain; iPTM matters for complexes). The raw
command:

Single chain:

```bash
python scripts/experiments/boltz_embedding_iptm_optimization.py \
    --sequence "MVTPEGNV...LYFQ" \
    --pdb_id 1u0p \
    --out_dir ./out_iptm_1u0p \
    --outer_steps 5 --inner_steps 1 --stop_opt_after_t 160 \
    --lr 1e-3 --batch_size 16
```

Multi-chain complex (iPTM is most informative here):

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

By default only `z_opt` is optimized; pass `--optimize_s` to also update
`s_opt`. The checkpoint must include `confidence_module` and
`distogram_module` or the script errors at startup.

---

## 4. Running NMR-restraint optimization

Unlike iPTM, this needs **two input files per target** — a reference structure
and a parsed NOE restraint CSV (same format as the Protenix fork), under
`nmr_pipeline_inputs/`:

```
nmr_pipeline_inputs/
├── pdbs/<ID>/<ID>_fixed.pdb     # heavy-atom reference  -> --nmr_reference_pdb
└── restraints/<ID>/<ID>.csv     # parsed NOE restraints -> --nmr_restraint_file
```

### Inputs (target `1u0p`)

The `1u0p` example is already provided at:

```
nmr_pipeline_inputs/pdbs/1u0p/1u0p_fixed.pdb
nmr_pipeline_inputs/restraints/1u0p/1u0p.csv
```

For another target, either **regenerate from the PDB ID** with the bundled
scraper (downloads the structure + NMR-STAR restraints from the RCSB, cleans
with PDBFixer, parses restraints → CSV):

```bash
python scripts/experiments/prepare_nmr_target.py <ID>   # writes into nmr_pipeline_inputs/
```

(needs `pynmrstar` + `pdbfixer` on top of the Boltz env — see the deps note in
§1), or **copy them from the Protenix fork** (format is identical):
`cp -r ../protenix/nmr_pipeline_inputs/{pdbs,restraints}/<ID> nmr_pipeline_inputs/…/`.

The restraint file is a **CSV** (the loss does `pd.read_csv`), not a `.tbl`.

### Run

**Ready-to-run example:** [`scripts/experiments/run_nmr_1u0p.sh`](scripts/experiments/run_nmr_1u0p.sh)
— `bash` it once the two files are in place; edit `SEQUENCE` / `PDB_ID` /
`NMR_*` at the top for another target. The raw command:

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

Loss stack (weighted sum):

| Term | Module | What it encodes |
| --- | --- | --- |
| `main_loss` | `NMRLossFunction` | NOE + optional methyl / amide RDC + relaxation restraints |
| `bond_length_loss` | `BondLengthLossFunction` | covalent geometry + clash penalty |
| `violation_loss` | `ViolationLossFunction` | OpenFold-style structural violations |
| `anchor_loss` | `AnchorLossFunction` | L2 regularization of `(s_opt, z_opt)` toward the pairformer baseline |

Each term's gradient norm w.r.t. both `s_opt` and `z_opt` is logged per
inner step — useful for diagnosing which objective is actually moving
which embedding.

Asset requirements:
- The fragment / name pickles are vendored in the package and resolve
  automatically (`src/boltz/experiments/nmr/assets/`). Override only if needed
  via `--nmr_assets_dir` or `$BOLTZ_NMR_ASSETS_DIR`.
- `--nmr_reference_pdb` is stripped of hydrogens and OXTs before loss
  computation so its topology matches Boltz heavy-atom output.
- Optional `--atom_index_map_npy` permutes Boltz atom order into the
  NMR loss's expected order.

---

## 5. Where the code lives

```
src/boltz/model/modules/diffusionv2.py     # potentials-guidance hook (lines 454–484)
src/boltz/experiments/
  nmr_loss_function.py                     # top-level NMR loss entry
  nmr/
    nmr_loss_function.py                   # NOE + RDC + relaxation losses
    bond_length_loss_function.py           # covalent geometry + clashes
    violation_loss_function.py             # OpenFold-style violations
    anchor_loss_function.py                # L2 to pairformer baseline
    s_2_loss_function.py                   # order-parameter loss (experimental)
    abstract_loss_funciton.py              # common base class
    openfold_violations/                   # adapted OpenFold geometry
    assets/                                # fragments.pickle, names.pickle
    hydrogen_addition.py, rmsd.py, utils_io.py, assets.py
scripts/experiments/
  boltz_embedding_iptm_optimization.py     # iPTM/pTM driver
  boltz_embedding_nmr_optimization.py      # NMR driver
  prepare_nmr_target.py                    # scrape & parse NMR inputs for one PDB id
  run_iptm_1u0p.sh                         # example iPTM run (edit SEQUENCE/PDB_ID)
  run_nmr_1u0p.sh                          # example NMR run (needs the two 1u0p files)
  template_diffusion_experiment.py         # adjacent: fixed-point work (not IT-opt)
  run_1u0p_modes.{sh,sbatch}               # SLURM sweep for fixed-point work
  run_all_fixed_point.sh                   # local sweep for fixed-point work
  README.md                                # detailed per-script reference
nmr_pipeline_inputs/               # per-target NMR inputs (ships 1u0p)
  pdbs/<ID>/<ID>_fixed.pdb                  # reference structure
  restraints/<ID>/<ID>.csv                 # parsed NOE restraints
```

The scripts import the model internals directly (`input_embedder`,
`msa_module`, `pairformer_module`, `structure_module`,
`confidence_module`, …) rather than going through `boltz predict`, so any
change to the in-fork Boltz package is picked up immediately.

---

## 6. Outputs and logging

Per outer loop `k`, both scripts write into `--out_dir`:

- `optimized_outer{k}.cif` — final denoised structure for sample 0.
- `embeddings_outer{k}.pt` — `{s_base, z_base, s_opt, z_opt}` on CPU.
- `coords_outer{k}.pt` — full batch of final atom coords (NMR script).
- `snapshot_outer{k}_step{step}.cif` — periodic trajectory snapshots
  (NMR script).

Running loss histories are checkpointed at the CWD:
- `losses_iptm_boltz_checkpoint.pt`
- `losses_nmr_boltz_checkpoint.pt`

Every inner step is logged to W&B. Default projects:
- `boltz-embedding-iptm-optim`
- `boltz-embedding-nmr-optim`

Use `--no_wandb` to disable, or `--wandb_project / --wandb_entity /
--wandb_run_name` to override. The saved `losses_*_checkpoint.pt` files hold the
full per-step loss/score histories if you want to plot them yourself.

---

## 7. Knobs worth tuning first

- `--outer_steps` — more noise realisations averages over diffusion
  variance. Start with 5 (iPTM) / 20 (NMR).
- `--inner_steps` — gradient steps per timestep on `(s_opt, z_opt)`. 1 is
  usually enough; larger values blow up compute and risk overfitting to
  that timestep.
- `--stop_opt_after_t` — freeze optimization once `t` exceeds this
  (counted from `T → 0`). Default 160 for both scripts; lower it if
  late-stage steps drift; raise it if you need more aggressive guidance.
- `--lr`, `--max_grad_norm` — Adam learning rate and gradient clip on
  `(s_opt, z_opt)`. Defaults: iPTM `1e-3 / 1.0`, NMR `0.025 / 0.6`.
- `--batch_size` — number of diffusion samples per outer step. Memory
  scales linearly.
- (NMR) `--lambda_*` — weights on the auxiliary terms. The violation /
  bond-length terms keep geometry sane; the anchor term keeps you from
  drifting off the pairformer manifold.
- (iPTM) `--optimize_s` — also gradient-step `s_opt`. Off by default
  because some kernels don't leave a gradient path through `s`.

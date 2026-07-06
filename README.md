# `it_opt/` — Inference-Time Optimization for Structure Prediction

This repository demonstrates the **same inference-time optimization (IT-opt)
idea across three different AlphaFold 3 reimplementations** ("remakes"):
**AlphaFold 3** itself, **Boltz-2**, and **Protenix**. The point is to show that
IT-opt is not specific to one codebase — because all three share the same
trunk → diffusion architecture, the same embedding-optimization recipe ports to
each. Each subdirectory is a self-contained fork (or snapshot) of the upstream
model with the IT-opt code added on top.

## The shared idea

All three models share the same backbone: a trunk (Evoformer / Pairformer)
produces single (`s`) and pair (`z`) embeddings, and a diffusion module turns
those embeddings into 3-D coordinates.

```
trunk(MSA, templates, …) ──► (s, z) ──► diffusion ──► coords ──► confidence head
                              ▲ frozen during standard inference
```

Standard inference freezes `(s, z)` and samples coordinates once. **IT-opt
instead treats `(s, z)` (or a structured perturbation of them) as trainable
parameters at inference time**, unrolls the differentiable diffusion sampler,
and runs gradient descent on the embeddings to optimize an objective evaluated
on the *denoised* structure — no retraining, no weight updates. After
optimization, a final sampling pass with the optimized embeddings produces the
structures that get saved.

The objective is task-dependent and is the main thing that differs between
experiments:

| Objective | What it does | Where |
|---|---|---|
| Confidence (`iPTM` / `pTM` / `pLDDT`) | push the model toward higher self-confidence; useful for complex docking | af3, boltz |
| NMR restraints | bias sampling toward NMR-consistent structures (NOE / RDC / relaxation) | boltz, protenix |
| Anchor / regularization | keep the optimized embeddings near the trunk baseline (added to any objective) | all three |

A common regularization theme across the forks is an **anchor (L2) term** that
keeps the optimized `(s, z)` from drifting off the learned manifold, plus
**early-stopping / freezing** late in the diffusion schedule to limit
distribution shift.

### Where the optimization loop lives in each fork

The heart of every fork is the same nested loop — *outer* (full diffusion
passes / fresh noise) → *diffusion timesteps* → *inner* (gradient steps on the
embeddings). Jump straight to it:

| Fork | Entry point | Outer loop | Inner loop |
|---|---|---|---|
| af3 | `optimize_embeddings()` in `af3/src/alphafold3/model/embedding_optimization.py:817` | `:1276` | `:1330` (+ diffusion loop `:1288`) |
| boltz | `main()` in `boltz/scripts/experiments/boltz_embedding_{iptm,nmr}_optimization.py` | iPTM `:514` / NMR `:624` | iPTM `:538` / NMR `:654` (+ diffusion loop `:519`/`:635`) |
| protenix | `run_it_optimization()` in `protenix/it_optimization_manager.py:288` | `:321` | per-timestep step `:335` (2-level: one Adam step per diffusion step) |

Each directory README expands on this with the backprop / Adam-step lines.

## Directory layout

```
it_opt/
├── af3/        AlphaFold 3 IT-opt (embedding optimization + coordinate guidance)
├── boltz/      Boltz-2 fork with embedding-space IT-opt (iPTM + NMR) and a
│               template↔diffusion fixed-point experiment
└── protenix/   Protenix fork with an IT-opt manager (NMR + anchor losses)
```

### `af3/` — AlphaFold 3

A **snapshot** of just the files needed to reproduce IT-opt of AlphaFold 3
trunk embeddings: two model modules (`embedding_optimization.py`,
`coordinate_guidance.py`) and the base AF3 checkpoint. These files are meant to
be dropped into a full AlphaFold 3 installation and driven through
`run_alphafold.py` CLI flags. See **[`af3/README.md`](af3/README.md)**.

| Path | Role |
|---|---|
| `src/alphafold3/model/embedding_optimization.py` | Core IT-opt: Adam updates of trunk `single`/`pair` embeddings against a differentiable iPTM/PTM/pLDDT objective, with structured update modes (low-rank, per-residue, channel-scale, …). |
| `src/alphafold3/model/coordinate_guidance.py` | Baseline: classifier-style guidance on the *denoised coordinates* instead of the embeddings. |
| `models/af3.bin.zst` | Base AlphaFold 3 checkpoint (zstd-compressed). |

### `boltz/` — Boltz-2

A full **fork of Boltz-2** (the upstream package lives under `src/boltz/`),
editable-installed, with IT-opt experiments added under `scripts/experiments/`
and custom loss modules under `src/boltz/experiments/`. See
**[`boltz/IT_Opt_Boltz.md`](boltz/IT_Opt_Boltz.md)** for the IT-opt quickstart
and **[`boltz/scripts/experiments/README.md`](boltz/scripts/experiments/README.md)**
for the per-script reference. (`boltz/README.md` is the upstream Boltz README.)

| Path | Role |
|---|---|
| `src/boltz/` | Upstream Boltz-2 package (model, modules, data). |
| `src/boltz/experiments/nmr/` | NMR loss stack (NOE/RDC/relaxation, OpenFold violations, bond-length, anchor). |
| `scripts/experiments/boltz_embedding_iptm_optimization.py` | Optimize `(s, z)` to maximize `0.8·iPTM + 0.2·pTM`. |
| `scripts/experiments/boltz_embedding_nmr_optimization.py` | Optimize `(s, z)` to minimize NMR-restraint + geometry losses. |
| `scripts/experiments/template_diffusion_experiment.py` | Template ↔ diffusion fixed-point experiment (does a predicted structure converge when fed back as a template?). |
| `scripts/{eval,process,train}/` | Upstream Boltz evaluation / data-processing / training utilities. |

### `protenix/` — Protenix

A **fork of Protenix** (upstream package under `src/protenix/`) with a
single-entry-point IT-opt driver. See **[`protenix/README.md`](protenix/README.md)**.

| Path | Role |
|---|---|
| `it_optimization_manager.py` | IT-opt driver: optimizes `(s_inputs, s_trunk, z_trunk)` through the diffusion process against the NMR loss. |
| `prepare_nmr_target.py` | Scrapes & parses NMR inputs for one PDB ID (download + clean + parse restraints → CSV). |
| `pipeline_configurations/*.yaml` | Run configs (sequence, model, optimizer, loss weights). |
| `src/losses/` | Loss modules: anchor, NMR, bond-length, violation. |
| `src/utils/non_diffusion_model_manager.py` | Wraps the Protenix model (trunk + diffusion) for stepwise, differentiable access. |
| `src/protenix/` | Upstream Protenix package. |
| `src/af3-dev/release_model/` | Protenix model checkpoint. |

## NMR data: `*/nmr_pipeline_inputs/`

The NMR-restraint optimization (in `boltz/` and `protenix/`) reads per-target
inputs from a `nmr_pipeline_inputs/` directory, keyed by PDB ID:

```
nmr_pipeline_inputs/
├── pdbs/<pdb_id>/<pdb_id>_fixed.pdb        # cleaned (e.g. PDBFixer'd) reference structure
└── restraints/<pdb_id>/<pdb_id>.csv        # parsed NOE restraints (one row per member)
```

One worked example target (`1u0p`) is bundled under
`protenix/nmr_pipeline_inputs/`. Any other target is prepared on demand from its
PDB ID with `protenix/prepare_nmr_target.py`, which scrapes the RCSB PDB
(structure + NMR-STAR restraints), cleans the structure with PDBFixer, and
parses the restraints into the CSV the loss reads. The Protenix README documents
the procedure and CSV schema: see
[`protenix/README.md`](protenix/README.md#preparing-a-target-scrape--parse). The
Boltz NMR driver expects the same per-target files (passed via
`--nmr_restraint_file` and `--nmr_reference_pdb`).

## Running

There is no single combined environment — each fork has its own model
dependencies. Start from the directory-specific README:

- AlphaFold 3 IT-opt → [`af3/README.md`](af3/README.md)
- Boltz-2 IT-opt → [`boltz/IT_Opt_Boltz.md`](boltz/IT_Opt_Boltz.md)
- Protenix IT-opt → [`protenix/README.md`](protenix/README.md)

## Attribution

These are derivative works built on three upstream projects, retained under
their respective licenses:

- **AlphaFold 3** — Google DeepMind
- **Boltz-1 / Boltz-2** — MIT License (see `boltz/LICENSE`)
- **Protenix** — ByteDance, CC BY-NC 4.0

## Citation

```
@inproceedings{maddipatla2026inference,
  title={Inference-time optimization for experiment-grounded protein ensemble generation},
  author={Maddipatla, Sai Advaith and Rzayev, Anar and Pegoraro, Marco and Pacesa, Martin and Schanda, Paul and Marx, Ailie and Vedula, Sanketh and Bronstein, Alexander},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
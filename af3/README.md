# `it_opt/` — Inference-Time Optimization of AlphaFold 3 Embeddings

This directory is a self-contained snapshot of the code and the base model
checkpoint needed to reproduce **inference-time optimization (IT-opt)** of
AlphaFold 3 trunk embeddings to maximize confidence scores (iPTM / PTM /
pLDDT). No experiment outputs are included.

These files are **extracted** from an AlphaFold 3 working tree — they are not a
full AF3 install. To run them, drop the two model modules into a complete
[AlphaFold 3](https://github.com/google-deepmind/alphafold3) checkout (see
[Running](#running) below).

## Contents

| Path | Role |
|---|---|
| `src/alphafold3/model/embedding_optimization.py` | Core IT-opt: per-step Adam updates of trunk `single`/`pair` embeddings against a differentiable iPTM/PTM/pLDDT objective. |
| `src/alphafold3/model/coordinate_guidance.py` | Sibling baseline: classifier-style guidance by ∇(confidence) w.r.t. *denoised coordinates* instead of embeddings. |
| `models/af3.bin.zst` | Base AlphaFold 3 checkpoint (zstd-compressed, byte-identical to upstream `models/af3.bin.zst`). |

IT-opt is driven entirely through `run_alphafold.py` CLI flags (see
[Running](#running)) — there are no launcher scripts in this snapshot. To sweep
over structures / perturbation / MSA-on-off, wrap the example invocation below
in your own shell or SLURM loop.

There is **no separate "MSA optimization" module** — MSA-on vs. MSA-off is just
a runtime flag (`--no_msa`) into the same embedding-optimization code path;
sweeping both regimes just means looping the flag over `true`/`false`.

## Model weights

This code runs on top of [AlphaFold 3](https://github.com/google-deepmind/alphafold3),
DeepMind's biomolecular structure-prediction model. **AlphaFold 3 model
parameters are access-gated by Google DeepMind and cannot be downloaded with a
public URL** (unlike Boltz/Protenix). You must request them yourself:

1. Request access by completing the form linked from the AlphaFold 3 repository
   (see its *"Obtaining Model Parameters"* section):
   <https://github.com/google-deepmind/alphafold3>. The terms prohibit
   redistributing the weights, so they are not bundled with this snapshot for
   sharing — obtain your own copy.
2. Once granted, you receive the `af3.bin` checkpoint (optionally
   zstd-compressed as `af3.bin.zst`; the runner reads either).

**Where to place it:** put the checkpoint in a directory and point
`--model_dir` at that directory — the runner looks for `af3.bin` /
`af3.bin.zst` inside it. The layout this snapshot assumes is:

```
af3/
└── models/
    └── af3.bin.zst        # <-- place the AlphaFold 3 checkpoint here
```

so runs pass `--model_dir="${AF3_DIR}/models"` (see [Running](#running)).

## How it works

### Embedding optimization (the main method)

Standard AlphaFold 3 inference is:

```
evoformer(MSA, templates, …) → (s_trunk, z_trunk) ──► diffusion → coords ──► confidence head → iPTM/PTM/pLDDT
                                  ▲
                                  └─ frozen during sampling
```

IT-opt unrolls the diffusion sampler and treats `(s_trunk, z_trunk)` —
or a structured perturbation of them — as **trainable parameters at inference
time**, optimized to maximize the confidence score on the denoised coordinates:

```
for outer_step in range(outer_steps):                    # full diffusion passes
  positions = initial_noise
  for diff_step in range(diffusion_steps):
    positions_noisy = augment_and_noise(positions, σ)
    if diff_step % optimization_interval == 0:
      for _ in range(optimization_steps):                # inner Adam loop
        grads, denoised = ∇_emb score(confidence_head(denoise(emb, positions_noisy, σ)))
        emb = adam_update(emb, grads)                    # in-place embedding update
    else:
      denoised = denoise(emb, positions_noisy, σ)
    positions = euler_update(positions_noisy, denoised, σ)
```

**Where the nested loop lives:** the whole thing is implemented in
`optimize_embeddings(...)` in `src/alphafold3/model/embedding_optimization.py`
(definition at line 817). The three loops are, concretely:

| Loop | Code | Line |
|---|---|---|
| Outer — full diffusion passes / noise realizations | `for outer_step in range(opt_config.outer_steps):` | `embedding_optimization.py:1276` |
| Middle — unrolled diffusion timesteps | `for step_idx in range(diffusion_steps):` | `embedding_optimization.py:1288` |
| Inner — Adam updates on the embeddings | `for opt_step in range(opt_config.optimization_steps):` | `embedding_optimization.py:1330` |

**Where the optimization actually happens** (inside that inner loop):

- **Objective + loss** — `step_loss_fn(...)` (`embedding_optimization.py:1088`)
  denoises the noisy positions and scores the result; the scalar loss
  `loss = -score + anchor_s + anchor_z` is formed at `embedding_optimization.py:1163`.
- **Gradient** — `grad_fn = jax.jit(jax.grad(step_loss_fn, ...))`
  (`embedding_optimization.py:1167–1168`), called for the embeddings at
  `embedding_optimization.py:1367` (or per-sample at `:1345` in the
  gradient-accumulation path).
- **Adam update** — `adam_update(...)` then `apply_update(...)` to rebuild the
  embeddings, at `embedding_optimization.py:1379–1384`.

A compact version of the whole skeleton is also in the module docstring at
`embedding_optimization.py:19–24`.

Crucially, the confidence head's distogram features use **soft (sigmoid) binning**
(`soft_dgram_from_positions`) during optimization so gradients flow through
distance-bin selection. After optimization, a final inference pass runs with
the optimized embeddings to produce the structures and scores that get saved.

The runner orchestrating this lives in `run_alphafold.py:760` —
`ModelRunner.run_inference_with_optimization` — which (1) does an initial
forward pass with `return_embeddings=True` to capture `s_trunk`/`z_trunk`,
(2) calls `embedding_optimization.optimize_embeddings(…)`, then (3) re-runs
the model with the optimized embeddings.

### Structured update modes

Rather than optimize the raw `(N, N, C)` pair tensor directly, `update_mode`
selects a constrained parameterization (see `EmbeddingOptimizationConfig` and
`apply_update` in `embedding_optimization.py`):

| Mode | Update | # extra params |
|---|---|---|
| `full` | `pair = pair` (raw tensor) | N·N·C |
| `low_rank` | `pair_init + Σ_r U_r ⊗ V_r` | 2·N·r·C |
| `low_rank_diagonal` | low-rank + diagonal residue terms | 2·N·r·C + N·C |
| `per_residue_bias` | `pair_init + left_i + right_j` | 2·N·C |
| `channel_scale` | `α_c · pair_init + β_c` | 2·C |
| `element_scale` | `(1 + β) · pair_init` (scalar) | 1 |
| `element_scale_channel` | `(1 + β_c) · pair_init` | C |
| `element_scale_residue` | `(1 + β_i β_j) · pair_init` | N |
| `element_scale_residue_channel` | `(1 + β_ic β_jc) · pair_init` | N·C |
| `zero_sum` | `pair_init + (A − mean_{ij} A)` | N·N·C |
| `zero_sum_channel` | `pair_init + (A − mean_c A)` | N·N·C |

`optimize_single` toggles whether `s_trunk` is also optimized (default off —
matches FKSFold-Boltz, which freezes `s_trunk`).

### Regularization / early-stopping

- **`anchor_lambda_single` / `anchor_lambda_pair`** — L2 anchor toward the
  initial embeddings, keeps the optimization from drifting off-manifold.
- **`perturbation_cliff_pct`** — hard cliff: if `‖Δemb‖ / ‖emb_init‖`
  exceeds this percentage, optimization stops and normal diffusion resumes.
- **`score_threshold`** — stop early once the score reaches this value
  (scores are bounded by 1.0, so 1.0 = disabled).
- **`cross_chain_only`** — only update inter-chain pair entries (focuses
  optimization on the interface; intra-chain stays frozen).
- **`pairformer_projection`** — optional periodic projection back onto the
  learned manifold by running Pairformer on the perturbed embeddings (with
  `stop_gradient`).

### Coordinate guidance (sibling method)

`coordinate_guidance.py` is the simpler baseline: it leaves the embeddings
alone and instead adds a drift term to the *denoised coordinates* at each
diffusion step, using ∇(score) w.r.t. those coordinates. It exposes
`guidance_scale`, `guidance_start_fraction` / `guidance_stop_fraction` to
restrict guidance to a window of the diffusion schedule, and (for
`scoring_type="contact"` / `"pae_proxy"`) direct geometric scores that bypass
the confidence head entirely.

## Running

These files are a **snapshot** — they are not on `PYTHONPATH` on their own.
To actually run IT-opt, place
`src/alphafold3/model/{embedding_optimization,coordinate_guidance}.py` into a
full AlphaFold 3 checkout, and make `run_alphafold.py` import them (around
lines 47–48) and expose the IT-opt CLI flags (see below). Then invoke the
standard runner with IT-opt flags — for example:

```bash
python run_alphafold.py \
  --fasta_path="${FASTA_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --model_dir="${AF3_DIR}/models" \
  --db_dir="${AF3_DIR}/databases" \
  --run_inference=true \
  --run_data_pipeline=false \
  --no_msa=${NO_MSA_FLAG} \
  --use_embedding_optimization=true \
  --emb_opt_outer_steps=10 \
  --emb_opt_optimization_steps=1 \
  --emb_opt_optimization_interval=1 \
  --emb_opt_batch_size=1 \
  --emb_opt_learning_rate=5e-3 \
  --emb_opt_anchor_lambda=0.0 \
  --emb_opt_perturbation_cliff_pct=0.01 \
  --emb_opt_score_threshold=1.0 \
  --emb_opt_scoring_type="0.8*iptm+0.2*ptm" \
  --emb_opt_update_mode=full \
  --evaluate_vs_gt=true \
  --gt_structure_path="${GT_STRUCTURE_PATH}"
```

To use coordinate guidance instead, swap
`--use_embedding_optimization=true` for `--use_coordinate_guidance=true`
and pass the `--coord_guidance_*` flags (defined in `run_alphafold.py:577+`).
The two flags are mutually exclusive.

### Key CLI flags (all defined in `run_alphafold.py:434–633`)

**Embedding optimization:**
`--use_embedding_optimization`,
`--emb_opt_outer_steps`,
`--emb_opt_optimization_steps`,
`--emb_opt_optimization_interval`,
`--emb_opt_diffusion_steps`,
`--emb_opt_learning_rate`,
`--emb_opt_anchor_lambda`,
`--emb_opt_scoring_type`,
`--emb_opt_max_grad_norm`,
`--emb_opt_optimize_single`,
`--emb_opt_update_mode`,
`--emb_opt_low_rank`,
`--emb_opt_batch_size`,
`--emb_opt_gradient_accumulation`,
`--emb_opt_perturbation_cliff_pct`,
`--emb_opt_score_threshold`,
`--emb_opt_cross_chain_only`,
`--emb_opt_pairformer_projection`,
`--emb_opt_pairformer_projection_interval`,
`--emb_opt_ipsae_pae_cutoff`,
`--emb_opt_ipsae_soft_temperature`.

**Coordinate guidance:**
`--use_coordinate_guidance`,
`--coord_guidance_scale`,
`--coord_guidance_diffusion_steps`,
`--coord_guidance_interval`,
`--coord_guidance_start_fraction`,
`--coord_guidance_stop_fraction`,
`--coord_guidance_scoring_type`,
`--coord_guidance_contact_threshold`,
`--coord_guidance_contact_temperature`,
`--coord_guidance_max_grad_norm`.

## Outputs produced by a run

Each run writes into `${OUTPUT_DIR}/${STRUCTURE}/`:

- `*_model.cif` — best predicted structure.
- `seed-*_sample-*/…_model.cif` — per-sample structures.
- `seed-*_embeddings/*_embeddings.npz` — the **optimized** embeddings.
- `*_data.json` — confidence scores (iPTM, PTM, pLDDT, PAE, …).
- Optimization trajectory CSVs (one row per inner step) when `output_dir`
  is passed to `optimize_embeddings`, recording score / loss / grad-norm /
  perturbation magnitude per step.

## Scoring types

The `scoring_type` string is parsed as a weighted linear combination, e.g.
`"0.8*iptm+0.2*ptm"`, `"iptm"`, `"plddt"`, or combinations with `ipsae`
(symmetric AE with PAE cutoff `--emb_opt_ipsae_pae_cutoff`). Coordinate
guidance additionally supports `"contact"` (differentiable interface contact
count) and `"pae_proxy"` (direct distance-based PAE-like score) that do not
go through the confidence head at all.

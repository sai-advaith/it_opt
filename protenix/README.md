# Protenix — Inference-Time Optimization

A fork of [Protenix](https://github.com/bytedance/Protenix) with an
**inference-time optimization (IT-opt)** driver. Rather than retraining, it
treats the trunk features `(s_inputs, s_trunk, z_trunk)` as optimization
variables and walks them through the (differentiable) diffusion process under a
configurable objective evaluated on the denoised structure.

## Contents

| Path | Role |
|---|---|
| `it_optimization_manager.py` | The IT-opt driver. Builds the model, runs the optimization loop over the diffusion process, and saves structures / metrics. Entry point for all runs. |
| `prepare_nmr_target.py` | Scrapes & parses NMR inputs for one PDB ID (download restraints + structure, clean, parse restraints → CSV). See [Preparing a target](#preparing-a-target-scrape--parse). |
| `pipeline_configurations/*.yaml` | Run configs: protein sequence + reference PDB, model-manager settings (recycles, diffusion steps, checkpoint path), optimizer hyperparameters, and the loss stack with its weights. |
| `nmr_pipeline_inputs/` | NMR loss inputs (`metadata/`, `pdbs/`, `restraints/` per PDB ID). Ships one worked example (`1u0p`); prepare others with `prepare_nmr_target.py`. See [The NMR restraint dataset](#the-nmr-restraint-dataset). |
| `src/losses/` | Loss modules: anchor (L2 to the trunk baseline), NMR (NOE/RDC/relaxation), bond-length + clash, OpenFold-style violations. |
| `src/utils/non_diffusion_model_manager.py` | `ProtenixModelManager` — wraps the model so the trunk and each diffusion step can be called individually and differentiably. Also handles MSA querying and structure I/O. |
| `src/utils/{io,mmseqs_query,pdb_parsing}.py` | Config loading, MSA-server queries, and PDB parsing helpers. |
| `src/runner/inference.py`, `src/configs/`, `src/protenix/` | Upstream Protenix inference runner, config defaults, and model package. |
| `src/af3-dev/release_model/` | Protenix model checkpoint (`model_v0.2.0.pt`). |
| `protenix_output/` | Default dump directory for processed inputs / intermediate model outputs. |

## How it works

The manager (`ITOptimizationManager`) loads the trunk features once, then for a
number of **outer diffusion passes** it optimizes `(s_trunk, z_trunk)` with Adam
against a weighted loss. The **main objective** is the NMR restraint loss
(`loss_function_type: "nmr"`): a NOE restraint-violation loss (+ optional
methyl/amide RDC and relaxation) that biases sampling toward NMR-consistent
structures. See [NMR-Restraint Optimization](#nmr-restraint-optimization).

An **anchor** term (L2 toward the trunk baseline, `anchor_loss_function`) is
always added on top of the main objective. The `bond_length` and `violation`
loss modules also exist and are used as geometry diagnostics. The learning
rate, number of outer steps, batch size, and gradient clipping are all set in
the YAML config.

**Where the nested loop lives:** `ITOptimizationManager.run_it_optimization()`
in `it_optimization_manager.py` (definition at line 288).

| Loop | Code | Line |
|---|---|---|
| Outer — full diffusion passes (`outer_diffusion_steps`) | `for diffusion_loop_iteration in range(num_full_diffusion_optimization_steps):` | `it_optimization_manager.py:321` |
| Inner — one optimization step per diffusion timestep | `for diffusion_step in diffusion_steps:` | `it_optimization_manager.py:335` |

**Where the optimization actually happens** (inside that inner loop):

- **Adam optimizer over the embeddings** — `torch.optim.Adam([optimized_s, optimized_z], lr=learning_rate)` at `it_optimization_manager.py:337` (rebuilt each step), `optimizer.zero_grad()` at `:340`.
- **Differentiable forward** — `denoise_net_batched(...)` → `it_optim_x0_hat` at `it_optimization_manager.py:345–354` (this is the call gradients flow through).
- **Loss** — `get_loss_values(...)` → `total_loss` at `it_optimization_manager.py:356` (main loss + anchor terms).
- **Backprop → clip → step** — `total_loss.backward()` (`:357`) → `clip_grad_norm_` (`:360–361`) → `optimizer.step()` (`:367`).

The actual diffusion step forward is then taken under `torch.no_grad()` at
`it_optimization_manager.py:372–392` — that part is *not* optimized.

Unlike the AlphaFold 3 and Boltz forks (which add a third, innermost
`optimization_steps` / `inner_steps` loop), Protenix takes exactly **one Adam
step per diffusion timestep**, so its loop nest is two levels deep rather than
three.

## Installation

### Environment Setup

1. **Setup the environment:**

   Create a fresh conda environment with Python 3.11:
   ```bash
   conda create -n guided_af3 python=3.11
   conda activate guided_af3
   ```
   Install the core scientific stack:
   ```bash
   pip3 install numpy==1.26.4 scipy==1.15.0 pandas==2.2.0 matplotlib==3.9.0 scikit-learn==1.2.0 scikit-learn-extra==0.3.0 skan==0.13.0 scikit-image==0.24.0 imageio==2.37.0 cvxpy==1.6.6 cvxpylayers==0.1.9
   ```
   Install bioinformatics and structure libraries:
   ```bash
   pip3 install biopython==1.83 biotite==1.0.1 gemmi==0.6.5 rdkit==2023.09.6 dm-tree==0.1.8 py3dmol==2.4.2 modelcif==0.7 loco-hd==0.1.4 pynmrstar==3.3.5 ml-collections==0.1.1
   ```
   Install utilities and logging:
   ```bash
   pip3 install tqdm pyyaml ipywidgets wandb==0.19.4 ipdb==0.13.13 icecream==2.1.4 hydride==1.2.3 pydantic==2.10.6 pdbeccdutils==0.8.5
   ```
   Install PyTorch with CUDA 12.1:
   ```bash
   pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
   ```
   Install AlphaFold-related JAX / TF packages (CPU-only here):
   ```bash
   pip3 install absl-py==1.0.0 dm-haiku==0.0.12 docker==5.0.0 jax==0.4.26 jaxlib==0.4.26 tensorflow-cpu==2.16.1 "pytest<8.5.0" "setuptools<72.0.0"
   ```
   Install Keops
   ```bash
   pip3 install pykeops==2.3 geomloss==0.2.6
   python3
   >>> import pykeops; pykeops.test_torch_bindings() # test keops install
   ```
   Install PDBFixer (https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html)
   ```bash
   git clone https://github.com/openmm/pdbfixer.git
   cd pdbfixer
   python setup.py install 
   ```

2. **Download model weights and data:**

   This pipeline is built on top of [Protenix](https://github.com/bytedance/Protenix), a PyTorch reproduction of DeepMind's AlphaFold3. Download the required pre-trained model weights and chemical component data files:

   ```bash
   # Download model weights (v0.2.0)
   wget -P src/af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt

   # Download chemical component dictionary files
   wget -P src/af3-dev/release_data/ https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif
   wget -P src/af3-dev/release_data/ https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl
   ```

   For more information, visit the Protenix repository: https://github.com/bytedance/Protenix

   **Where to place them:** the `wget -P` commands above already download into
   the directories the code expects. The resulting layout (relative to this
   `protenix/` directory) is:

   ```
   src/af3-dev/
   ├── release_model/
   │   └── model_v0.2.0.pt                              # <-- model checkpoint
   └── release_data/
       ├── components.v20240608.cif                     # CCD dictionary
       └── components.v20240608.cif.rdkit_mol.pkl       # parsed RDKit mols
   ```

   The checkpoint path is wired into the run configs as
   `model_checkpoint_path: "./src/af3-dev/release_model/model_v0.2.0.pt"` (see
   `pipeline_configurations/nmr_baseline.yaml`); change it there if you place
   the weights elsewhere.

## Usage

### NMR-Restraint Optimization

This mode biases sampling toward structures consistent with **experimental NMR
distance restraints (NOEs)**. The
objective is the NOE restraint-violation loss (`NMRLossFunction`), optionally
combined with methyl/amide RDC and relaxation terms, plus the anchor term that
keeps the optimized `(s, z)` near the trunk baseline.

Select it in the config with `loss_function_type: "nmr"` — see
[`pipeline_configurations/nmr_baseline.yaml`](pipeline_configurations/nmr_baseline.yaml).

#### The NMR restraint dataset

`nmr_pipeline_inputs/` holds the per-target inputs, keyed by PDB ID:

```
nmr_pipeline_inputs/
├── metadata/<ID>/<ID>.json     # {"pdb_id", "seq"} — the chain sequence
├── pdbs/<ID>/<ID>.pdb          # deposited NMR structure
│              <ID>_fixed.pdb   # cleaned (PDBFixer'd) reference — used by the loss
└── restraints/<ID>/<ID>.str    # original NMR-STAR restraint file (source)
                    <ID>.csv    # parsed NOE restraints — what the loss reads
```

Only **`1u0p`** is bundled here as a worked example (the one the config uses).
For each target the loss needs two files: the **reference structure**
(`<ID>_fixed.pdb` — the loss adds hydrogens internally, so heavy atoms suffice)
and the **parsed NOE restraint CSV** (`<ID>.csv`). Prepare any other target
with the script below.

#### Preparing a target (scrape & parse)

[`prepare_nmr_target.py`](prepare_nmr_target.py) turns a PDB ID into the files
above, in one command:

```bash
python prepare_nmr_target.py 1u0p        # writes into nmr_pipeline_inputs/ by default
```

What it does (see the module docstring for details):

1. **Download restraints** — fetches the NMR-STAR restraint file from the RCSB
   PDB (`https://files.rcsb.org/download/<ID>_mr.str`, with `.str.gz` / `.mr`
   fallbacks) → `restraints/<ID>/<ID>.str`.
2. **Download structure** — `https://files.rcsb.org/download/<ID>.pdb` →
   `pdbs/<ID>/<ID>.pdb`.
3. **Clean structure** — first model, hydrogens removed, missing atoms rebuilt
   with PDBFixer → `pdbs/<ID>/<ID>_fixed.pdb`.
4. **Record metadata** — the chain sequence → `metadata/<ID>/<ID>.json`.
5. **Parse restraints** — reads the NMR-STAR `_Gen_dist_constraint` loop with
   `pynmrstar`, maps each proton to its parent heavy atom, and writes the flat
   NOE CSV (schema below) → `restraints/<ID>/<ID>.csv`.

Requires `requests`, `pandas`, `pynmrstar`, `gemmi`, and `openmm`/`pdbfixer`
(all part of the Protenix environment above). Not every PDB entry has deposited
restraints — the script reports and exits if none are found. Heavy-atom
assignment uses a name-based heuristic; spot-check unusual residues.

After preparing a target, update the config (`protein.sequence` from the printed
sequence, and the `reference_pdb` / `pdb_file` / `reference_nmr` paths) and run.

#### Restraint CSV schema

`NMRLossFunction` reads the CSV and keeps rows where `type == "NOE"`. Columns:

| Column | Meaning |
|---|---|
| `type` | restraint type; only `NOE` rows are used |
| `constrain_id` | restraint id; rows sharing an id are OR-combined (ambiguous restraint) |
| `member_id` | member index within an ambiguous restraint |
| `member_logic` | member combination logic (`OR` / `AND` / `.`) |
| `residue1_num`, `residue1_id` | residue number and 3-letter name for atom 1 |
| `atom1`, `heavy_atom1` | atom-1 name and its parent heavy atom |
| `residue2_num`, `residue2_id` | residue number and name for atom 2 |
| `atom2`, `heavy_atom2` | atom-2 name and parent heavy atom |
| `distance` | target distance (Å) |
| `lower_bound`, `upper_bound` | distance bounds (Å); `lower_bound` may be `.` (treated as 0) |

Example:

```csv
type,constrain_id,member_id,member_logic,residue1_num,residue1_id,atom1,heavy_atom1,residue2_num,residue2_id,atom2,heavy_atom2,distance,lower_bound,upper_bound
NOE,1,1,OR,8,HIS,H,N,7,HIS,HB2,CB,1.8,1.8,4.30
NOE,2,1,.,8,HIS,HA,CA,9,SER,H,N,1.8,1.8,2.88
```

For multi-chain targets, add optional `chain1`/`chain2` columns; restraints
where `chain1 != chain2` are treated as inter-chain.

#### Run

Point the config at your target's files — `protein.reference_pdb`,
`nmr_loss_function.pdb_file` (both the `<ID>_fixed.pdb`) and
`nmr_loss_function.reference_nmr` (the `<ID>.csv`) — then:

```
python3 it_optimization_manager.py -- configuration_file pipeline_configurations/nmr_baseline.yaml
```

#### NMR configuration reference

Fields specific to the NMR run (beyond the shared `general` / `protein` /
`model_manager` / `it_optimization_parameters` blocks):

| Field | Meaning |
|---|---|
| `loss_function.loss_function_type: "nmr"` | selects the NMR objective (currently the only supported main loss) |
| `loss_function.nmr_loss_function.reference_nmr` | path to the parsed NOE CSV (`restraint_file` also accepted) |
| `loss_function.nmr_loss_function.pdb_file` | reference structure for the loss (falls back to `reference_pdb`, then `protein.reference_pdb`) |
| `loss_function.nmr_loss_function.noe_scale` | weight on the NOE term (tune per target — the NMR loss magnitude differs from RMSD) |
| `loss_function.nmr_loss_function.iid_loss` | per-restraint vs OR-group reduction |
| `loss_function.nmr_loss_function.{methyl,amide}_{rdc,relax}_{file,scale}` | optional RDC / relaxation terms (`null` + `0.0` ⇒ NOE-only) |
| `loss_function.anchor_loss_function.{lambda_s,lambda_z}` | anchor (L2 to trunk baseline) weights — always applied on top of the main loss |

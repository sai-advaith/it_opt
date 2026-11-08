# ChimeraX-ITopt

Geometric restraints for IT-opt (inference-time optimization of Protenix trunk
embeddings, github.com/sai-advaith/it_opt), defined interactively in ChimeraX.

## Setup (tested on macOS, Apple M4 Max, ChimeraX 1.10)
1. Protenix weights + CCD data as in `../protenix/README.md` (`src/af3-dev/release_model`, `release_data`).
2. Backend env (PyTorch with MPS/CUDA):
   ```
   conda create -n itopt python=3.11
   conda activate itopt
   pip install torch numpy==1.26.4 scipy pandas matplotlib biopython==1.83 biotite==1.0.1 gemmi==0.6.5 \
       rdkit==2023.09.6 dm-tree ml-collections==0.1.1 tqdm pyyaml wandb requests pydantic \
       pdbeccdutils==0.8.5 modelcif==0.7 icecream scikit-learn
   ```
3. Install the bundle: `ChimeraX --nogui --exit --cmd "devel install /path/to/it_opt/chimerax exit true"`
4. In ChimeraX: `itopt config python <env>/bin/python protenix /path/to/it_opt/protenix`
   (defaults: `~/miniforge3/envs/itopt/bin/python`, `~/Documents/code/ext/it_opt/protenix`).
Panel: Tools → Structure Prediction → ITopt Restraints.

## Sketch workflow (recommended)
1. `itopt sketch #1` — editable copy (plum); your model is never modified.
2. Select a domain/loop on the sketch; right-drag translates it, shift-right-drag rotates it
   (about the centre of rotation; `cofr sel` to pivot on the selection).
3. `itopt sketch restrain` — each moved segment becomes a restraint against the unmoved part:
   rigid moves of >= 12 residues → orientation restraint, otherwise → CA position restraint.
4. `itopt run` — guided samples stream in as "ITopt iterNNN", then "ITopt final".

## Commands
| command | meaning |
|---|---|
| `itopt distance A B [target d] [tolerance t] [lower l] [upper u]` | 2 atoms, or groups `atoms with atoms` (centroids) |
| `itopt position ATOMS frame FRAME [to MARKERS] [tolerance t]` | pin atoms (or their centroid) in the frame of FRAME |
| `itopt substructure ATOMS [reference #N] [tolerance rmsd]` | keep a fragment's shape |
| `itopt orientation D1 with D2 [angleTol deg] [distTol A]` | relative domain pose |
| `itopt mouse distance` / `itopt mouse off` | right-click two atoms → distance restraint |
| `itopt list / delete ID… / save f.json / load f.json` | manage |
| `itopt run [samples 4] [outer 5] [steps 100] [cycles 4] [msa true] [lr 0.06]` | start job |
| `itopt stop / status / wait`, `itopt ghosts false`, `itopt config python … protenix … runsDir …` | |

Geometry is taken from wherever the atoms were picked (sketch or target), or from
`reference #N` (matched by chain/residue number/atom name).
Restraint colors: yellow pending, green satisfied in ≥50% of samples, red otherwise.

## Backend
`src/backend/run_restraints.py spec.json` runs in the `itopt` conda env
against the it_opt Protenix fork (`../protenix`); it does not import `src/losses` (that package currently fails to import: `s_2_loss_function` is missing); outputs in `~/itopt_runs/<name>/`:
`baseline_s*.cif`, `iter*_s*.cif` (guided), `final_s*.cif` (unguided resample from optimized
embeddings), `metrics.json`, `status.json`. MSAs (ColabFold server) and trunk embeddings are
cached in `~/.cache/itopt-chimerax`.

## Example: calmodulin (`examples/calmodulin`)
C-lobe rotated 60 deg on the sketch -> orientation restraint (<= 15 deg). 148 aa, MSA, 4 samples x 5 outer x 100 steps:
10 min on an M4 Max. Orientation error: baseline 45-55 deg; guided samples 8-29 deg (best 8 deg, image);
unguided resampling from the optimised embeddings 21-85 deg.

Note: the guided samples (`iter*`) are the ones that satisfy restraints; unguided resampling
from the optimized embeddings (`final`) did not retain them in early tests.

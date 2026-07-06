from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from boltz.data.types import StructureV2


@dataclass(frozen=True)
class ResolvedRestraint:
    """A restraint resolved to atom indices in a StructureV2."""

    chain1: str
    chain2: str
    atom1_idx: int
    atom2_idx: int
    lower: float
    upper: float


def _normalize_atom_name(atom: str, resname: str | None = None) -> str:
    """Map hydrogen-like atom names to a reasonable heavy-atom proxy.

    This is a pragmatic adapter to avoid needing explicit hydrogen placement.
    It is intended for restraint files that reference atoms like HB2/HD11/QD/MD1, etc.

    Strategy:
    - If the atom is already heavy (does not start with 'H'/'Q'/'M'), return as-is.
    - If atom begins with 'Q' or 'M' (group pseudo-hydrogens), map to the corresponding heavy carbon.
    - If atom begins with 'H', map common patterns (HA->CA, HB*->CB, HG*->CG, HD*->CD, HE*->CE, HZ*->CZ/NZ).
    """

    a = atom.strip().upper()
    rn = (resname or "").strip().upper()

    # Group pseudo-hydrogens: map to heavy atoms typically used as anchors.
    if a.startswith("M"):  # methyl groups in many NMR tables
        # LEU MD1/MD2 -> CD1/CD2; VAL MG1/MG2 -> CG1/CG2; ILE MG/MD -> CG2/CD1, etc.
        mapping = {
            ("LEU", "MD1"): "CD1",
            ("LEU", "MD2"): "CD2",
            ("VAL", "MG1"): "CG1",
            ("VAL", "MG2"): "CG2",
            ("ILE", "MG"): "CG2",
            ("ILE", "MD"): "CD1",
            ("ALA", "MB"): "CB",
            ("MET", "ME"): "CE",
            ("THR", "MG"): "CG2",
        }
        return mapping.get((rn, a), a)

    if a.startswith("Q"):  # aromatic/amine pseudo-hydrogens
        mapping = {
            ("TYR", "QD"): "CD1",
            ("TYR", "QE"): "CE1",
            ("PHE", "QD"): "CD1",
            ("PHE", "QE"): "CE1",
            ("TRP", "QD"): "CD1",
            ("TRP", "QE"): "NE1",
            ("TRP", "QZ"): "CZ2",
            ("LYS", "QZ"): "NZ",
        }
        return mapping.get((rn, a), a)

    # If not hydrogen-like, keep.
    if not a.startswith("H"):
        return a

    # Backbone
    if a in {"H", "HN", "H1", "H2", "H3", "HN1", "HN2"}:
        return "N"
    if a.startswith("HA"):
        return "CA"

    # Sidechain: HB*, HG*, HD*, HE*, HZ*
    if a.startswith("HB"):
        return "CB"
    if a.startswith("HG"):
        return "CG"
    if a.startswith("HD"):
        return "CD"
    if a.startswith("HE"):
        return "CE"
    if a.startswith("HZ"):
        # Lys HZ* -> NZ (amine), aromatic HZ -> CZ
        return "NZ" if rn == "LYS" else "CZ"

    # Fallback: keep as-is; it may already match an atom name in the structure.
    return a


def _build_atom_index(structure: StructureV2) -> dict[tuple[str, int, str], int]:
    """Build (chain_name, resnum_1idx, atom_name) -> atom_idx mapping.

    Boltz `StructureV2` does not store PDB residue numbers, only internal ordering.
    Here we treat residue number as *1-indexed position within the chain*.
    """

    index: dict[tuple[str, int, str], int] = {}
    atoms = structure.atoms
    residues = structure.residues
    chains = structure.chains

    for chain in chains:
        chain_name = str(chain["name"])
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for pos0, res in enumerate(residues[res_start:res_end]):
            resnum = pos0 + 1  # 1-index within chain
            atom_start = int(res["atom_idx"])
            atom_end = atom_start + int(res["atom_num"])
            for aidx in range(atom_start, atom_end):
                atom_name = str(atoms[aidx]["name"]).strip().upper()
                index[(chain_name, resnum, atom_name)] = aidx

    return index


class NMRNOELossFunction:
    """A lightweight NOE restraint loss for Boltz structures.

    Reads a restraint CSV in the same general format as `msa_subsampling`:
    - required columns: residue1_num, residue1_id, atom1, residue2_num, residue2_id, atom2, lower_bound, upper_bound, type
    - optional: chain1, chain2

    Loss:
    - hinge on bounds: relu(d - ub) + relu(lb - d)
    - averaged across restraints

    Notes:
    - Residue numbers are interpreted as 1-indexed positions within each chain (Boltz internal order).
    - Hydrogen/pseudo-atom names are mapped to heavy atom proxies via `_normalize_atom_name()`.
    """

    def __init__(
        self,
        restraint_file: str | Path,
        structure: StructureV2,
        device: torch.device,
        noe_scale: float = 1.0,
    ) -> None:
        self.device = device
        self.noe_scale = float(noe_scale)
        self.structure = structure
        self.atom_index = _build_atom_index(structure)

        df = pd.read_csv(restraint_file)
        if "type" in df.columns:
            df = df[df["type"].astype(str).str.upper() == "NOE"].reset_index(drop=True)
        self.df = df

        # Normalize bounds ('.' -> 0 for lower)
        lower = df["lower_bound"].apply(lambda x: 0.0 if str(x).strip() == "." else float(x)).to_numpy()
        upper = df["upper_bound"].astype(float).to_numpy()
        self.lower = torch.tensor(lower, dtype=torch.float32, device=device)
        self.upper = torch.tensor(upper, dtype=torch.float32, device=device)

        # Resolve to atom indices once
        self.resolved: list[ResolvedRestraint] = []
        for i, row in df.iterrows():
            c1 = str(row["chain1"]) if "chain1" in df.columns else str(structure.chains[0]["name"])
            c2 = str(row["chain2"]) if "chain2" in df.columns else c1
            r1 = int(row["residue1_num"])
            r2 = int(row["residue2_num"])
            rn1 = str(row.get("residue1_id", "")).upper()
            rn2 = str(row.get("residue2_id", "")).upper()
            a1 = _normalize_atom_name(str(row["atom1"]), rn1)
            a2 = _normalize_atom_name(str(row["atom2"]), rn2)

            key1 = (c1, r1, a1)
            key2 = (c2, r2, a2)
            if key1 not in self.atom_index or key2 not in self.atom_index:
                # skip unresolvable rows
                continue
            self.resolved.append(
                ResolvedRestraint(
                    chain1=c1,
                    chain2=c2,
                    atom1_idx=self.atom_index[key1],
                    atom2_idx=self.atom_index[key2],
                    lower=float(lower[i]),
                    upper=float(upper[i]),
                )
            )

        self.last_loss: Optional[float] = None
        self.num_restraints = len(self.resolved)

    def __call__(self, atom_coords: torch.Tensor) -> torch.Tensor:
        """Compute NOE hinge loss.

        Parameters
        ----------
        atom_coords : Tensor
            Shape (B, N_atoms, 3) in the same atom ordering as `structure.atoms`.
        """
        if self.num_restraints == 0:
            out = torch.tensor(0.0, device=self.device)
            self.last_loss = float(out.item())
            return out

        # Gather coords for each restraint.
        idx1 = torch.tensor([r.atom1_idx for r in self.resolved], device=self.device, dtype=torch.long)
        idx2 = torch.tensor([r.atom2_idx for r in self.resolved], device=self.device, dtype=torch.long)
        p1 = atom_coords[:, idx1, :]
        p2 = atom_coords[:, idx2, :]
        d = torch.linalg.vector_norm(p1 - p2, dim=-1)  # (B, R)

        lower = torch.tensor([r.lower for r in self.resolved], device=self.device, dtype=torch.float32)[None, :]
        upper = torch.tensor([r.upper for r in self.resolved], device=self.device, dtype=torch.float32)[None, :]

        ub = torch.relu(d - upper)
        lb = torch.relu(lower - d)
        loss = (ub + lb).mean() * self.noe_scale
        self.last_loss = float(loss.item())
        return loss

    def wandb_log(self) -> dict[str, Any]:
        return {"noe_loss": self.last_loss, "noe_num_restraints": self.num_restraints}


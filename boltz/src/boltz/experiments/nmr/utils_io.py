"""IO helpers used by the vendored NMR loss stack.

This file is intentionally minimal: the ported `NMRLossFunction` only needs
`load_pdb_atom_locations()` and the `AMINO_ACID_ATOMS_ORDER` table.
"""

from __future__ import annotations

import gemmi
import numpy as np
import torch


AMINO_ACID_ATOMS_ORDER = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "MSE": ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "CSO": ["N", "CA", "C", "O", "CB", "SG", "OD"],
}


def load_pdb_atom_locations(pdb_file: str, device: str = "cpu", single_model: bool = True) -> torch.Tensor:
    """Load atom coordinates from a PDB in a consistent per-residue atom order."""
    structure = gemmi.read_structure(pdb_file)
    structure.remove_hydrogens()

    models_to_process = [structure[0]] if single_model else structure
    all_models_positions = []

    for model in models_to_process:
        model_positions = []
        chain = model[0]  # Assumes first chain is relevant
        for residue in chain:
            atoms_order = AMINO_ACID_ATOMS_ORDER.get(residue.name)
            if atoms_order is None:
                continue
            try:
                residue_atoms = [
                    [atom for atom in residue if atom.name == atom_name][0] for atom_name in atoms_order
                ]
            except IndexError:
                continue
            for atom in residue_atoms:
                model_positions.append((atom.pos.x, atom.pos.y, atom.pos.z))

        if model_positions:
            all_models_positions.append(model_positions)

    atom_positions_array = np.array(all_models_positions)
    return torch.tensor(atom_positions_array, dtype=torch.float32, device=device)


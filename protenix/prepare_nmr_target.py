"""
Scrape and prepare NMR restraint inputs for a single target.

Given a PDB ID, this:
  1. Downloads the NMR restraint file (NMR-STAR ``<ID>_mr.str``) from the RCSB PDB.
  2. Downloads the deposited structure (``<ID>.pdb``) from the RCSB PDB.
  3. Cleans it (first model, hydrogens removed, missing atoms rebuilt with
     PDBFixer) -> ``<ID>_fixed.pdb``.
  4. Records the chain sequence in ``metadata/<ID>/<ID>.json``.
  5. Parses the NMR-STAR distance-constraint loop with ``pynmrstar`` into the
     flat NOE CSV the loss reads -> ``restraints/<ID>/<ID>.csv``.

Output layout (under --input_directory, default ``nmr_pipeline_inputs``):

    nmr_pipeline_inputs/
    ├── metadata/<ID>/<ID>.json
    ├── pdbs/<ID>/<ID>.pdb   and  <ID>_fixed.pdb
    └── restraints/<ID>/<ID>.str  and  <ID>.csv

Usage:
    python prepare_nmr_target.py 1u0p
    python prepare_nmr_target.py 2lk2 --input_directory nmr_pipeline_inputs

Then point pipeline_configurations/nmr_baseline.yaml at the produced files
(protein.sequence / reference_pdb and nmr_loss_function.reference_nmr / pdb_file)
and run it_optimization_manager.py.

Dependencies: requests, pandas, pynmrstar, gemmi, openmm + pdbfixer (all part of
the Protenix environment described in the README).

Adapted from the msa_subsampling NMR preprocessing pipeline. Heavy-atom
assignment here uses the self-contained regular-expression heuristic
(``guess_heavy_atom``); the original optionally used a per-residue bond graph.
"""
import argparse
import json
import os
import re

import gemmi
import pandas as pd
import pynmrstar
import requests
from openmm.app import PDBFile
import pdbfixer

aa_map = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'SEC': 'U', 'PYL': 'O',
    'MSE': 'M', 'CSO': 'C', 'HEM': '%', 'HOH': '&', 'IMD': '^',
}


def download_nmr_restraints(pdb_id, save_dir):
    """Download the NMR-STAR restraint file for ``pdb_id`` from the RCSB PDB.

    Tries the uncompressed ``<ID>_mr.str``, then the gzip, then the legacy
    ``<ID>.mr``. Saves to ``restraints/<id>/<id>.str``. Returns the path or None.
    """
    pdb_id = pdb_id.lower()
    nmr_restraints_dir = os.path.join(save_dir, "restraints", pdb_id)
    os.makedirs(nmr_restraints_dir, exist_ok=True)
    out = os.path.join(nmr_restraints_dir, f"{pdb_id}.str")

    urls = [
        f"https://files.rcsb.org/download/{pdb_id.upper()}_mr.str",
        f"https://files.rcsb.org/download/{pdb_id.upper()}_mr.str.gz",
        f"https://files.rcsb.org/download/{pdb_id.upper()}.mr",
    ]
    for url in urls:
        try:
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            dst = out + ".gz" if url.endswith(".gz") else out
            with open(dst, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded restraints for {pdb_id.upper()} from {url}")
            return dst
        except requests.exceptions.RequestException as e:
            print(f"  {url} -> {e}")
    print(f"No restraint file found for {pdb_id.upper()} "
          "(the entry may not have deposited restraints).")
    return None


def download_pdb_file(pdb_id, save_dir):
    """Download the deposited PDB structure. Returns the path or None."""
    pdb_id = pdb_id.lower()
    nmr_pdbs_dir = os.path.join(save_dir, "pdbs", pdb_id)
    os.makedirs(nmr_pdbs_dir, exist_ok=True)
    out = os.path.join(nmr_pdbs_dir, f"{pdb_id}.pdb")
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(out, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded structure for {pdb_id.upper()} to {out}")
        return out
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDB {pdb_id.upper()}: {e}")
        return None


def fix_pdb(pdb_file_path, pdb_id):
    """First model, hydrogens removed, missing atoms rebuilt with PDBFixer.

    Writes ``<ID>_fixed.pdb`` next to the input and returns its path.
    """
    dir_name = os.path.dirname(pdb_file_path)
    structure = gemmi.read_structure(pdb_file_path)
    source_model = structure[0]
    source_model.remove_hydrogens()

    new = gemmi.Structure()
    new.name = structure.name
    new.spacegroup_hm = structure.spacegroup_hm
    new.cell = structure.cell
    dst_model = gemmi.Model(source_model.name if source_model.name else "1")
    for src_chain in source_model:
        dst_chain = gemmi.Chain(src_chain.name)
        for src_res in src_chain:
            dst_res = gemmi.Residue()
            dst_res.name = src_res.name
            dst_res.seqid = src_res.seqid
            dst_res.het_flag = src_res.het_flag
            for src_atom in src_res:
                dst_atom = gemmi.Atom()
                dst_atom.name = src_atom.name
                dst_atom.element = src_atom.element
                dst_atom.occ = src_atom.occ
                dst_atom.b_iso = src_atom.b_iso
                dst_atom.pos = src_atom.pos
                dst_atom.charge = src_atom.charge
                dst_atom.altloc = src_atom.altloc
                if src_atom.aniso is not None:
                    dst_atom.aniso = src_atom.aniso
                dst_res.add_atom(dst_atom)
            dst_chain.add_residue(dst_res)
        dst_model.add_chain(dst_chain)
    new.add_model(dst_model)

    new_pdb_file_path = os.path.join(dir_name, f"{pdb_id.lower()}_fixed.pdb")
    new.setup_entities()
    new.write_pdb(new_pdb_file_path)

    fixer = pdbfixer.PDBFixer(filename=new_pdb_file_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    PDBFile.writeFile(fixer.topology, fixer.positions, open(new_pdb_file_path, "w"))
    return new_pdb_file_path


def guess_heavy_atom(hydrogen_name, aa=None):
    """Heuristic parent-heavy-atom name for an NMR hydrogen/proton name.

    Handles backbone amide (H/HN -> N), alpha (HA* -> CA), old XPLOR methyl
    names (MG1 -> CG, ...), and side-chain protons (HG12 -> CG1, HD21 -> CD2).
    ``aa`` is accepted for API compatibility but not used by the heuristic.
    """
    if hydrogen_name in ['H', 'HN', 'H1', 'H2', 'H3']:
        return 'N'
    if hydrogen_name.startswith('HA'):
        return 'CA'
    m = re.match(r'^M([A-Z])\d*$', hydrogen_name)
    if m:
        return 'C' + m.group(1)
    m = re.match(r'^H([A-Z0-9]+)(\d)$', hydrogen_name)
    if m:
        return 'C' + m.group(1)
    if hydrogen_name.startswith('H'):
        return 'C' + hydrogen_name[1:]
    return None


def extract_distance_restraints(star_file_path, verbose=False):
    """Parse the NMR-STAR ``_Gen_dist_constraint`` loop into the NOE CSV schema."""
    try:
        entry = pynmrstar.Entry.from_file(star_file_path)
    except Exception as e:
        print(f"Error loading {star_file_path}: {e}")
        return pd.DataFrame()

    rows = []
    for saveframe in entry:
        if 'distance_constraints' not in saveframe.category.lower():
            continue
        try:
            constraint_type = saveframe.get_tag('_Gen_dist_constraint_list.Constraint_type')[0]
        except Exception:
            constraint_type = 'Unknown'

        for loop in saveframe:
            if loop.category != '_Gen_dist_constraint':
                continue
            idx = {tag: loop.tags.index(tag) for tag in (
                'ID', 'Seq_ID_1', 'Comp_ID_1', 'Atom_ID_1',
                'Seq_ID_2', 'Comp_ID_2', 'Atom_ID_2',
                'Distance_val', 'Member_ID', 'Member_logic_code',
                'Distance_lower_bound_val', 'Distance_upper_bound_val')}
            for row in loop.data:
                comp1, atom1 = row[idx['Comp_ID_1']], row[idx['Atom_ID_1']]
                comp2, atom2 = row[idx['Comp_ID_2']], row[idx['Atom_ID_2']]
                if comp1 == "THR" and atom1 == "HH":
                    atom1 = "HG1"
                if comp2 == "THR" and atom2 == "HH":
                    atom2 = "HG1"
                heavy1 = guess_heavy_atom(atom1, comp1) if atom1[0] in "HMQ" else atom1
                heavy2 = guess_heavy_atom(atom2, comp2) if atom2[0] in "HMQ" else atom2
                rows.append({
                    'type': constraint_type,
                    'constrain_id': row[idx['ID']],
                    'member_id': row[idx['Member_ID']],
                    'member_logic': row[idx['Member_logic_code']],
                    'residue1_num': row[idx['Seq_ID_1']],
                    'residue1_id': comp1,
                    'atom1': atom1,
                    'heavy_atom1': heavy1,
                    'residue2_num': row[idx['Seq_ID_2']],
                    'residue2_id': comp2,
                    'atom2': atom2,
                    'heavy_atom2': heavy2,
                    'distance': row[idx['Distance_val']],
                    'lower_bound': row[idx['Distance_lower_bound_val']],
                    'upper_bound': row[idx['Distance_upper_bound_val']],
                })
    return pd.DataFrame(rows)


def get_amino_acid_sequence(pdb_file_path):
    return "".join(aa_map[res.name] for res in gemmi.read_pdb(pdb_file_path)[0][0])


def save_metadata(pdb_id, sequence, root):
    metadata_dir = os.path.join(root, "metadata", pdb_id.lower())
    os.makedirs(metadata_dir, exist_ok=True)
    with open(os.path.join(metadata_dir, f"{pdb_id.lower()}.json"), "w") as f:
        json.dump({"pdb_id": pdb_id, "seq": sequence}, f, indent=4)


def main(pdb_id, input_directory):
    pdb_id = pdb_id.lower()
    restraint_str = download_nmr_restraints(pdb_id, input_directory)
    if restraint_str is None:
        raise SystemExit(f"No restraints available for {pdb_id}; aborting.")
    pdb_path = download_pdb_file(pdb_id, input_directory)
    if pdb_path is None:
        raise SystemExit(f"Could not download structure for {pdb_id}; aborting.")

    fixed_pdb = fix_pdb(pdb_path, pdb_id)
    sequence = get_amino_acid_sequence(fixed_pdb)
    save_metadata(pdb_id, sequence, input_directory)

    df = extract_distance_restraints(restraint_str)
    csv_path = os.path.join(input_directory, "restraints", pdb_id, f"{pdb_id}.csv")
    if df.shape[0] == 0:
        raise SystemExit(f"No distance restraints parsed for {pdb_id}.")
    print(f"Parsed {df.shape[0]} restraints for {pdb_id}:")
    print(df.groupby('type').size())
    df.to_csv(csv_path, index=False)

    print("\nPrepared:")
    print(f"  sequence         : {sequence}")
    print(f"  reference (fixed): {fixed_pdb}")
    print(f"  restraints (csv) : {csv_path}")
    print(f"  metadata         : {input_directory}/metadata/{pdb_id}/{pdb_id}.json")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape & prepare NMR restraints for one PDB target.")
    parser.add_argument("pdb_id", type=str, help="4-character PDB ID (e.g. 1u0p)")
    parser.add_argument("--input_directory", type=str, default="nmr_pipeline_inputs",
                        help="Root output directory (default: nmr_pipeline_inputs)")
    args = parser.parse_args()
    main(args.pdb_id, args.input_directory)

"""Write a Protenix atom array + coordinates as mmCIF with user chain ids/numbering."""

import gemmi
import numpy as np


def write_cif(atom_array, coords, chain_labels, res_numbers, path, title="itopt"):
    """res_numbers: {label: [author resnum for seq pos 1..L] or None}."""
    model_chains = list(dict.fromkeys(atom_array.chain_id.tolist()))
    label_of = dict(zip(model_chains, chain_labels))
    st = gemmi.Structure()
    st.name = title
    model = gemmi.Model("1")
    chains = {}
    cur_key, res = None, None
    for i in range(len(atom_array)):
        a = atom_array[i]
        lab = label_of[a.chain_id]
        if lab not in chains:
            chains[lab] = gemmi.Chain(lab)
        key = (lab, int(a.res_id))
        if key != cur_key:
            if res is not None:
                chains[cur_key[0]].add_residue(res)
            res = gemmi.Residue()
            res.name = str(a.res_name)
            nums = res_numbers.get(lab)
            seqpos = int(a.res_id)
            res.seqid = gemmi.SeqId(int(nums[seqpos - 1]) if nums else seqpos, " ")
            res.label_seq = seqpos
            res.entity_type = gemmi.EntityType.Polymer
            cur_key = key
        at = gemmi.Atom()
        at.name = str(a.atom_name)
        at.element = gemmi.Element(str(a.element) if getattr(a, "element", "") else str(a.atom_name)[0])
        at.pos = gemmi.Position(*map(float, coords[i]))
        at.occ = 1.0
        at.b_iso = 0.0
        res.add_atom(at)
    if res is not None:
        chains[cur_key[0]].add_residue(res)
    for lab in chain_labels:
        if lab in chains:
            model.add_chain(chains[lab])
    st.add_model(model)
    st.setup_entities()
    for ent in st.entities:  # write entity_poly_seq so viewers know the full sequence
        if ent.entity_type == gemmi.EntityType.Polymer:
            sub = ent.subchains[0]
            ent.full_sequence = [r.name for c in st[0] for r in c if r.subchain == sub]
            ent.polymer_type = gemmi.PolymerType.PeptideL
    st.make_mmcif_document().write_file(path)

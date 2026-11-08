"""Session state: target structure, restraint list, display, spec export."""

import json
import os

import numpy as np

PB_GROUP = "ITopt restraints"
COLORS = {
    "pending": (255, 200, 0, 255),
    "ok": (0, 200, 80, 255),
    "bad": (230, 40, 40, 255),
}
RESULT_COLORS = {"baseline": "gray", "iter": "orange", "final": "cornflowerblue"}


def get_state(session, create=True):
    st = getattr(session, "_itopt_state", None)
    if st is None and create:
        st = ITOptState(session)
        session._itopt_state = st
    return st


class UserError(Exception):
    pass


def _user_error(msg):
    from chimerax.core.errors import UserError as CXUserError
    raise CXUserError(msg)


class ITOptState:
    def __init__(self, session):
        self.session = session
        self.target = None  # AtomicStructure
        self.restraints = []  # dicts; keys starting with "_" are ChimeraX-only
        self._counter = {}
        self.job = None
        self.frame_atoms = None  # panel helpers
        self.domain1_atoms = None
        self.sketch = None  # editable copy of the target; restraints read coords from it
        self.show_ghosts = True
        self._saved_bindings = None
        self.listeners = []  # callables() on change

    # ------------------------------------------------------------------ util
    def changed(self):
        for f in list(self.listeners):
            try:
                f()
            except Exception:
                pass

    def next_id(self, prefix):
        n = self._counter.get(prefix, 0) + 1
        self._counter[prefix] = n
        return f"{prefix}{n}"

    def set_target(self, structure):
        if self.target is not None and structure is not self.target and self.restraints:
            _user_error(f"Restraints already defined on {self.target}; 'itopt delete all' first")
        self.target = structure
        self.changed()

    def has_sketch(self):
        return self.sketch is not None and not self.sketch.deleted

    def resolve(self, atoms):
        """Map atoms picked on the target or on the sketch to target atoms.

        Returns (target_atoms, scene_coords) where the coordinates come from the
        model the atoms were picked on (so the sketch defines the geometry).
        """
        from chimerax.atomic import Atoms
        structs = atoms.unique_structures
        if len(structs) != 1:
            _user_error("Atoms must all be in one structure (the target or its sketch)")
        if self.has_sketch() and structs[0] is self.sketch:
            xyz = atoms.scene_coords
            tgt = [self.atom_from_key(self.atom_key(a)) for a in atoms]
            if any(t is None for t in tgt):
                _user_error("Sketch atom without a counterpart in the target")
            return Atoms(tgt), xyz
        self.ensure_target(atoms)
        return atoms, atoms.scene_coords

    def ensure_target(self, atoms):
        structs = atoms.unique_structures
        if len(structs) != 1:
            _user_error("Restrained atoms must all be in one structure (the prediction target)")
        s = structs[0]
        if self.target is None or self.target.deleted:
            self.target = s
        elif s is not self.target:
            _user_error(f"Atoms are in {s} but the IT-opt target is {self.target} "
                        "(use 'itopt target' or a 'reference' model instead)")
        return s

    # ----------------------------------------------------------- atom keys
    @staticmethod
    def heavy(atoms):
        return atoms.filter(atoms.element_names != "H")

    def atom_key(self, a):
        r = a.residue
        ch = r.chain
        if ch is None:
            _user_error(f"{a} is not in a polymer chain")
        return [r.chain_id, int(ch.res_map[r]) + 1, a.name]

    def keys(self, atoms):
        return [self.atom_key(a) for a in atoms]

    def atom_from_key(self, key):
        cid, pos, name = key
        for ch in self.target.chains:
            if ch.chain_id == cid:
                r = ch.residues[pos - 1]
                if r is None:
                    return None
                return r.find_atom(name)
        return None

    def reference_coords(self, atoms, reference):
        """Scene coords of the atoms matching `atoms` (by chain/resnum/name) in `reference`."""
        if reference is None:
            return atoms.scene_coords
        lut = {}
        for ra in reference.atoms:
            r = ra.residue
            lut[(r.chain_id, r.number, r.insertion_code, ra.name)] = ra
        out = []
        for a in atoms:
            r = a.residue
            ra = lut.get((r.chain_id, r.number, r.insertion_code, a.name))
            if ra is None:
                _user_error(f"No atom matching {a} in reference {reference}")
            out.append(ra.scene_coord)
        return np.array(out)

    # --------------------------------------------------------- chains/spec
    def chain_specs(self):
        from chimerax.atomic import Residue
        if self.target is None:
            _user_error("No IT-opt target structure; add a restraint or use 'itopt target'")
        out = []
        for ch in self.target.chains:
            if ch.polymer_type != Residue.PT_AMINO:
                continue
            seq = ch.characters
            if "X" in seq or "?" in seq:
                _user_error(f"Chain {ch.chain_id} has non-standard residues ({seq.count('X')} X); "
                            "only standard protein chains are supported for now")
            nums = [r.number if r is not None else None for r in ch.residues]
            out.append({"id": ch.chain_id, "sequence": seq, "res_numbers": _fill_numbers(nums)})
        if not out:
            _user_error(f"{self.target} has no protein chains")
        return out

    def spec(self, settings=None, output_dir=None, name="itopt"):
        rs = [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.restraints]
        return {
            "name": name,
            "output_dir": output_dir,
            "target": self.target.atomspec if self.target else None,
            "chains": self.chain_specs(),
            "restraints": rs,
            "settings": settings or {},
        }

    # ------------------------------------------------------------ restraints
    def add(self, r, display_atoms):
        r["_atoms"] = display_atoms
        r["_status"] = "pending"
        self.restraints.append(r)
        self.draw()
        self.changed()
        return r

    def delete(self, ids):
        if ids == "all":
            self.restraints = []
            self._counter = {}
        else:
            want = set(ids)
            missing = want - {r["id"] for r in self.restraints}
            if missing:
                _user_error(f"No restraint(s) {', '.join(sorted(missing))}")
            self.restraints = [r for r in self.restraints if r["id"] not in want]
        self.draw()
        self.changed()

    def describe(self, r):
        t = r["type"]
        a = r.get("_atoms", {})
        if t == "distance":
            lo, hi = r.get("lower"), r.get("upper")
            rng = f"{lo:.1f}-{hi:.1f} A" if lo is not None and hi is not None else (
                f"<= {hi:.1f} A" if hi is not None else f">= {lo:.1f} A")
            return f"{_spec(a.get('a'))} -- {_spec(a.get('b'))}: {rng}"
        if t == "position":
            return (f"{len(r['atoms'])} atoms -> {len(r['targets'])} target point(s) "
                    f"within {r['tolerance']:.1f} A, frame {len(r['frame']['atoms'])} atoms")
        if t == "substructure":
            return f"{_spec(a.get('atoms'))}: RMSD <= {r['tolerance']:.1f} A ({len(r['atoms'])} atoms)"
        if t == "orientation":
            return (f"{_spec(a.get('d1'))} vs {_spec(a.get('d2'))}: "
                    f"angle <= {r['angle_tol']:.0f} deg, shift <= {r['dist_tol']:.1f} A")
        return t

    # ------------------------------------------------------------- display
    def draw(self):
        s = self.target
        if s is None or s.deleted:
            return
        pbg = s.pseudobond_group(PB_GROUP, create_type="per coordset")
        pbg.clear()
        pbg.dashes = 6
        for r in self.restraints:
            if r["type"] == "distance":
                a, b = r["_atoms"]["a"], r["_atoms"]["b"]
                pb = pbg.new_pseudobond(a[0], b[0])
                pb.color = COLORS[r.get("_status", "pending")]
                pb.radius = 0.25
                r["_pb"] = pb
        self._draw_markers()

    def _draw_markers(self):
        from chimerax.markers import MarkerSet
        ms = getattr(self, "_markers", None)
        if ms is not None and not ms.deleted:
            ms.delete()
        self._markers = None
        pts = [(r, p) for r in self.restraints if r["type"] == "position" for p in r["_target_scene"]]
        traces = []  # (restraint, coords, rgba) ghost CA traces of the target geometry
        if self.show_ghosts:
            for r in self.restraints:
                c = COLORS[r.get("_status", "pending")]
                if r["type"] == "substructure":
                    traces.append((r, np.array(r["coords"]), c))
                elif r["type"] == "orientation":
                    traces.append((r, np.array(r["domain1"]["coords"]), (160, 160, 160, 255)))
                    traces.append((r, np.array(r["domain2"]["coords"]), c))
                elif r["type"] == "position" and len(r["targets"]) > 1:
                    traces.append((r, np.array(r["targets"]), c))
        if not pts and not traces:
            return
        ms = MarkerSet(self.session, "ITopt targets")
        self.session.models.add([ms])
        for r, p in pts:
            if len(r["targets"]) == 1 or not self.show_ghosts:
                ms.create_marker(p, COLORS[r.get("_status", "pending")], 0.6)
        from chimerax.markers import create_link
        for r, xyz, color in traces:
            prev = None
            for p in xyz:
                m = ms.create_marker(p, color, 0.35)
                if prev is not None and np.linalg.norm(p - prev.coord) < 4.5:
                    create_link(prev, m, rgba=color, radius=0.15)
                prev = m
        if traces:
            # orientation: link domain centroids so the relative pose is visible
            for r in self.restraints:
                if r["type"] == "orientation" and self.show_ghosts:
                    c1 = ms.create_marker(np.mean(r["domain1"]["coords"], 0), (160, 160, 160, 255), 1.0)
                    c2 = ms.create_marker(np.mean(r["domain2"]["coords"], 0),
                                          COLORS[r.get("_status", "pending")], 1.0)
                    create_link(c1, c2, rgba=(200, 120, 255, 255), radius=0.3)
        self._markers = ms

    def color_status(self, report, final=False):
        """report: list of {'id','satisfied': [bool]*B, ...} from the backend."""
        by_id = {x["id"]: x for x in (report or [])}
        for r in self.restraints:
            x = by_id.get(r["id"])
            if x is None:
                continue
            frac = sum(x["satisfied"]) / max(1, len(x["satisfied"]))
            r["_status"] = "ok" if frac >= 0.5 else "bad"
            r["_value"] = x["value_mean"]
            r["_frac"] = frac
            pb = r.get("_pb")
            if pb is not None and not pb.deleted:
                pb.color = COLORS[r["_status"]]
        if final:
            self._draw_markers()
        self.changed()

    # -------------------------------------------------------- persistence
    def save(self, path, settings=None):
        d = self.spec(settings=settings)
        with open(path, "w") as f:
            json.dump(d, f, indent=1)

    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        if self.target is None:
            _user_error("Set the target first with 'itopt target <structure>'")
        mine = {c["id"]: c["sequence"] for c in self.chain_specs()}
        for c in d["chains"]:
            if mine.get(c["id"]) != c["sequence"]:
                _user_error(f"Chain {c['id']} sequence in {path} differs from {self.target}")
        from chimerax.atomic import Atoms
        for r in d["restraints"]:
            def A(keys):
                atoms = [self.atom_from_key(k) for k in keys]
                return Atoms([a for a in atoms if a is not None])
            disp = {}
            if r["type"] == "distance":
                disp = {"a": A(r["a"]), "b": A(r["b"])}
            elif r["type"] == "position":
                disp = {"atoms": A(r["atoms"])}
                r_scene = np.array(r["targets"])
                r["_target_scene"] = r_scene
            elif r["type"] == "substructure":
                disp = {"atoms": A(r["atoms"])}
            elif r["type"] == "orientation":
                disp = {"d1": A(r["domain1"]["atoms"]), "d2": A(r["domain2"]["atoms"])}
            prefix = r["id"].rstrip("0123456789")
            num = int(r["id"][len(prefix):] or 0)
            self._counter[prefix] = max(self._counter.get(prefix, 0), num)
            self.add(r, disp)
        return len(d["restraints"]), d.get("settings", {})

    # ------------------------------------------------- superposition atoms
    def alignment_keys(self):
        keys = []
        for r in self.restraints:
            if r["type"] == "position":
                keys += r["frame"]["atoms"]
            elif r["type"] == "orientation":
                keys += r["domain1"]["atoms"]
            elif r["type"] == "substructure":
                keys += r["atoms"]
        if not keys and self.target is not None:
            keys = [self.atom_key(a) for a in self.target.atoms if a.name == "CA" and a.residue.chain is not None]
        seen, out = set(), []
        for k in keys:
            t = tuple(k)
            if t not in seen:
                seen.add(t)
                out.append(k)
        return out


def _spec(atoms):
    if atoms is None or len(atoms) == 0:
        return "?"
    if len(atoms) == 1:
        return atoms[0].string(style="command")
    from chimerax.atomic import concise_residue_spec
    try:
        spec = concise_residue_spec(atoms[0].structure.session, atoms.unique_residues)
        names = set(atoms.names)
        return spec + (f"@{names.pop()}" if len(names) == 1 else f" ({len(atoms)} atoms)")
    except Exception:
        return f"{len(atoms)} atoms"


def _fill_numbers(nums):
    """Fill author numbers for missing residues by extrapolating from neighbours."""
    n = len(nums)
    known = [i for i, v in enumerate(nums) if v is not None]
    if not known:
        return list(range(1, n + 1))
    out = list(nums)
    for i in range(n):
        if out[i] is None:
            j = min(known, key=lambda k: abs(k - i))
            out[i] = nums[j] + (i - j)
    # guarantee uniqueness (gaps in numbering can collide after extrapolation)
    if len(set(out)) != n:
        return list(range(1, n + 1))
    return out

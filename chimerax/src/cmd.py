"""`itopt` commands."""

import os
import time

import numpy as np
from chimerax.core.commands import (BoolArg, CmdDesc, EnumOf, FloatArg, IntArg, OpenFileNameArg,
                                    RestOfLine, SaveFileNameArg, StringArg, register)
from chimerax.core.errors import UserError
from chimerax.atomic import AtomsArg, StructureArg

from .state import get_state

SUBSET = EnumOf(("ca", "backbone", "heavy", "given"))
BACKBONE = {"N", "CA", "C", "O"}


def _subset(atoms, use):
    atoms = atoms.filter(atoms.element_names != "H")
    if use == "ca":
        atoms = atoms.filter(atoms.names == "CA")
    elif use == "backbone":
        atoms = atoms.filter(np.isin(atoms.names, list(BACKBONE)))
    if len(atoms) == 0:
        raise UserError(f"No atoms left after restricting to '{use}'")
    return atoms


def _log_added(session, r):
    st = get_state(session)
    session.logger.info(f"IT-opt restraint {r['id']} ({r['type']}): {st.describe(r)}")


# --------------------------------------------------------------------------- #
def itopt_target(session, structure):
    st = get_state(session)
    st.set_target(structure)
    chains = st.chain_specs()
    session.logger.info(f"IT-opt target {structure}: " +
                        ", ".join(f"chain {c['id']} ({len(c['sequence'])} aa)" for c in chains))


def _geom(st, atoms, reference):
    """target atoms + coordinates (sketch / picked model, or reference model)."""
    tgt, xyz = st.resolve(atoms)
    if reference is not None:
        xyz = st.reference_coords(tgt, reference)
    return tgt, np.asarray(xyz)


def itopt_distance(session, atoms, with_=None, target=None, tolerance=1.0, lower=None, upper=None,
                   weight=1.0, reference=None):
    st = get_state(session)
    atoms = st.heavy(atoms)
    if with_ is None:
        if len(atoms) != 2:
            raise UserError(f"Give exactly two atoms (got {len(atoms)}), or two groups with 'with'")
        a, b = atoms[0:1], atoms[1:2]
    else:
        a, b = atoms, st.heavy(with_)
    a, xa = _geom(st, a, reference)
    b, xb = _geom(st, b, reference)
    if target is None and lower is None and upper is None:
        target = float(np.linalg.norm(xa.mean(0) - xb.mean(0)))
    if target is not None:
        lower = max(0.0, target - tolerance) if lower is None else lower
        upper = target + tolerance if upper is None else upper
    r = {"id": st.next_id("d"), "type": "distance", "a": st.keys(a), "b": st.keys(b),
         "lower": lower, "upper": upper, "weight": weight}
    st.add(r, {"a": a, "b": b})
    _log_added(session, r)
    return r


def itopt_position(session, atoms, frame=None, to=None, tolerance=1.0, weight=1.0, use="given",
                   frame_use="ca", reference=None):
    st = get_state(session)
    atoms, xyz = _geom(st, _subset(atoms, use), reference)
    frame, frame_xyz = _geom(st, _subset(frame, frame_use), reference)
    if len(frame) < 3:
        raise UserError("The frame needs at least 3 atoms")
    if len(atoms & frame):
        raise UserError("Pinned atoms overlap the frame atoms")
    targets = xyz
    if to is not None:
        targets = to.scene_coords
        if len(targets) not in (1, len(atoms)):
            raise UserError(f"'to' has {len(targets)} points; need 1 (centroid) or {len(atoms)}")
    r = {"id": st.next_id("p"), "type": "position", "atoms": st.keys(atoms),
         "targets": np.asarray(targets).tolist(),
         "frame": {"atoms": st.keys(frame), "coords": frame_xyz.tolist()},
         "tolerance": tolerance, "weight": weight}
    r["_target_scene"] = np.asarray(targets)
    st.add(r, {"atoms": atoms})
    _log_added(session, r)
    return r


def itopt_substructure(session, atoms, reference=None, tolerance=1.0, weight=1.0, use="ca"):
    st = get_state(session)
    atoms, xyz = _geom(st, _subset(atoms, use), reference)
    if len(atoms) < 3:
        raise UserError("A substructure needs at least 3 atoms")
    r = {"id": st.next_id("s"), "type": "substructure", "atoms": st.keys(atoms),
         "coords": xyz.tolist(), "tolerance": tolerance, "weight": weight}
    st.add(r, {"atoms": atoms})
    _log_added(session, r)
    return r


def itopt_orientation(session, domain1, with_=None, reference=None, angle_tol=10.0, dist_tol=2.0,
                      weight=1.0, use="ca"):
    st = get_state(session)
    if with_ is None:
        raise UserError("Usage: itopt orientation <domain1> with <domain2>")
    d1, x1 = _geom(st, _subset(domain1, use), reference)
    d2, x2 = _geom(st, _subset(with_, use), reference)
    if len(d1) < 3 or len(d2) < 3:
        raise UserError("Each domain needs at least 3 atoms")
    if len(d1 & d2):
        raise UserError("Domains overlap")
    r = {"id": st.next_id("o"), "type": "orientation",
         "domain1": {"atoms": st.keys(d1), "coords": x1.tolist()},
         "domain2": {"atoms": st.keys(d2), "coords": x2.tolist()},
         "angle_tol": angle_tol, "dist_tol": dist_tol, "weight": weight}
    st.add(r, {"d1": d1, "d2": d2})
    _log_added(session, r)
    return r


# --------------------------------------------------------------- sketch mode
def itopt_sketch(session, structure=None):
    from chimerax.core.commands import run
    st = get_state(session)
    if structure is not None:
        st.set_target(structure)
    if st.target is None or st.target.deleted:
        raise UserError("No target: 'itopt sketch <structure>'")
    if st.has_sketch():
        session.logger.info(f"Sketch already open: {st.sketch}")
        return
    sk = st.target.copy("ITopt sketch")
    session.models.add([sk])
    st.sketch = sk
    run(session, f"color #{sk.id_string} plum target c; transparency #{sk.id_string} 30 target c", log=False)
    from .mouse import bind_sketch_modes
    bind_sketch_modes(session)
    session.logger.info(
        f"Sketch #{sk.id_string} created from {st.target} (your model is not modified).<br>"
        "Select a domain/loop on the sketch (ctrl-click; up-arrow extends to SSE/chain), then "
        "<b>right-drag</b> to translate it and <b>shift-right-drag</b> to rotate it "
        "(rotation is about the centre of rotation, e.g. <i>cofr sel</i>). "
        "When done: <b>itopt sketch restrain</b> (or the panel button) turns every moved segment into "
        "restraints; restraint commands given on sketch atoms also take their geometry from the sketch.",
        is_html=True)
    st.changed()


def itopt_sketch_close(session):
    st = get_state(session)
    from .mouse import restore_modes
    restore_modes(session)
    if st.has_sketch():
        session.models.close([st.sketch])
    st.sketch = None
    st.changed()


def _segments(flags, max_gap=1):
    """Contiguous runs of True (gaps of <= max_gap False are bridged)."""
    runs, cur, gap = [], [], 0
    for i, f in enumerate(flags):
        if f:
            cur.append(i)
            gap = 0
        elif cur:
            gap += 1
            if gap > max_gap:
                runs.append(cur)
                cur, gap = [], 0
    if cur:
        runs.append(cur)
    return runs


def itopt_sketch_restrain(session, threshold=1.0, tolerance=1.5, rigid_tol=1.0, min_domain=12,
                          angle_tol=10.0, weight=1.0):
    """Turn every segment moved in the sketch into a restraint.

    Moved = CA displaced by > threshold A from the target.  A moved segment of
    >= min_domain residues that moved rigidly (internal CA RMSD <= rigid_tol)
    becomes an orientation restraint against the unmoved part; anything else
    becomes a position restraint on its CA atoms in the frame of the unmoved part.
    """
    from chimerax.atomic import Atoms
    from chimerax.geometry import align_points
    st = get_state(session)
    if not st.has_sketch():
        raise UserError("No sketch: 'itopt sketch' first")
    sk, tg = st.sketch, st.target
    made = []
    pairs = []  # (chain_id, target CA, sketch CA) in sequence order
    for tch, sch in zip(tg.chains, sk.chains):
        for tr, sr in zip(tch.residues, sch.residues):
            if tr is None or sr is None:
                continue
            ta, sa = tr.find_atom("CA"), sr.find_atom("CA")
            if ta is not None and sa is not None:
                pairs.append((tch.chain_id, ta, sa))
    if not pairs:
        raise UserError("No CA atoms to compare")
    t_xyz = np.array([p[1].scene_coord for p in pairs])
    s_xyz = np.array([p[2].scene_coord for p in pairs])
    moved = np.linalg.norm(t_xyz - s_xyz, axis=1) > threshold
    if not moved.any():
        raise UserError(f"Nothing in the sketch moved by more than {threshold} A")
    if moved.all():
        raise UserError("Everything moved; keep part of the sketch fixed as the reference frame")
    # segments per chain (short unmoved gaps inside a moved segment, e.g. a hinge pivot, are bridged)
    idx_by_chain = {}
    for i, p in enumerate(pairs):
        idx_by_chain.setdefault(p[0], []).append(i)
    segments = []
    for cid, idx in idx_by_chain.items():
        for seg in _segments([moved[i] for i in idx], max_gap=2):
            segments.append([idx[k] for k in range(seg[0], seg[-1] + 1)])
    in_seg = set(i for seg in segments for i in seg)
    frame_sk = Atoms([pairs[i][2] for i in range(len(pairs)) if i not in in_seg])
    if len(frame_sk) < 3:
        raise UserError("Fewer than 3 unmoved residues left as the reference frame")
    for ii in segments:
        seg_sk = Atoms([pairs[i][2] for i in ii])
        rigid_rms = None
        if len(ii) >= 3:
            _, rigid_rms = align_points(t_xyz[ii], s_xyz[ii])
        if len(ii) >= min_domain and rigid_rms is not None and rigid_rms <= rigid_tol:
            r = itopt_orientation(session, frame_sk, with_=seg_sk, angle_tol=angle_tol,
                                  dist_tol=tolerance, weight=weight)
        else:
            r = itopt_position(session, seg_sk, frame=frame_sk, tolerance=tolerance, weight=weight)
        made.append(r["id"])
    session.logger.info(f"IT-opt: {len(made)} restraint(s) from sketch moves: {', '.join(made)} "
                        f"(frame = {len(frame_sk)} unmoved CA atoms)")


def itopt_ghosts(session, show=True):
    st = get_state(session)
    st.show_ghosts = show
    st.draw()


def itopt_mouse(session, mode="distance", button="right"):
    from .mouse import bind_distance_mode, restore_modes, bind_sketch_modes
    if mode == "distance":
        bind_distance_mode(session, button)
    elif mode == "sketch":
        bind_sketch_modes(session)
    else:
        restore_modes(session)


def itopt_list(session):
    st = get_state(session)
    if not st.restraints:
        session.logger.info("No IT-opt restraints")
        return
    lines = [f"IT-opt target {st.target}:"]
    for r in st.restraints:
        extra = ""
        if "_value" in r:
            extra = f"  [last: {r['_value']:.2f}, {100 * r['_frac']:.0f}% samples satisfied]"
        lines.append(f"  {r['id']:4s} {r['type']:12s} w={r.get('weight', 1)}  {st.describe(r)}{extra}")
    session.logger.info("\n".join(lines))


def itopt_delete(session, ids):
    ids = ids.strip()
    if not ids:
        raise UserError("Usage: itopt delete <id ...> | all")
    get_state(session).delete("all" if ids == "all" else ids.replace(",", " ").split())


def itopt_save(session, file_name):
    get_state(session).save(file_name)
    session.logger.info(f"Saved IT-opt restraints to {file_name}")


def itopt_load(session, file_name, target=None):
    st = get_state(session)
    if target is not None:
        st.set_target(target)
    n, _ = st.load(file_name)
    session.logger.info(f"Loaded {n} IT-opt restraints from {file_name}")


def itopt_run(session, samples=4, outer=5, steps=100, cycles=4, lr=0.06, msa=True, device="mps",
              lambda_s=1e-4, lambda_z=1e-3, baseline=True, guide_from=0, seed=0, name=None,
              restraint_scale=1.0):
    from .job import Job, settings
    st = get_state(session)
    if st.job is not None and not st.job.finished:
        raise UserError("An IT-opt job is already running ('itopt stop' first)")
    if not st.restraints:
        raise UserError("No restraints defined")
    cs = settings(session)
    if not os.path.exists(cs.python):
        raise UserError(f"Backend python not found: {cs.python} (set with 'itopt config python <path>')")
    name = name or f"{st.target.name.split('.')[0]}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(os.path.expanduser(cs.runs_dir), name)
    os.makedirs(out_dir, exist_ok=True)
    S = {"samples": samples, "outer": outer, "diffusion_steps": steps, "n_cycle": cycles, "lr": lr,
         "use_msa": msa, "device": device, "lambda_s": lambda_s, "lambda_z": lambda_z,
         "baseline": baseline, "guide_from_step": guide_from, "seed": seed,
         "restraint_scale": restraint_scale}
    spec = st.spec(settings=S, output_dir=out_dir, name=name)
    if msa:
        session.logger.info("IT-opt: MSAs are fetched from the public ColabFold server (api.colabfold.com) "
                            "and cached per sequence; use 'msa false' for single-sequence mode.")
    st.job = Job(session, spec, out_dir)
    st.changed()


def itopt_stop(session):
    st = get_state(session)
    if st.job is None or st.job.finished:
        raise UserError("No IT-opt job running")
    st.job.stop()


def itopt_wait(session, timeout=None):
    st = get_state(session)
    if st.job is None:
        raise UserError("No IT-opt job")
    st.job.wait(timeout)


def itopt_status(session):
    st = get_state(session)
    if st.job is None:
        session.logger.info("No IT-opt job")
        return
    st.job.poll()
    s = st.job.status
    session.logger.info(f"IT-opt job {st.job.out_dir}: {s.get('stage')} "
                        f"(outer {s.get('outer', 0) + 1}/{s.get('outers', '?')}, step {s.get('step', '?')})")


def itopt_config(session, python=None, protenix=None, runs_dir=None):
    from .job import settings
    cs = settings(session)
    if python:
        cs.python = os.path.expanduser(python)
    if protenix:
        cs.protenix = os.path.expanduser(protenix)
    if runs_dir:
        cs.runs_dir = os.path.expanduser(runs_dir)
    session.logger.info(f"IT-opt config: python={cs.python}\n  protenix={cs.protenix}\n  runs_dir={cs.runs_dir}")


# --------------------------------------------------------------------------- #
def register_commands(logger):
    W = ("weight", FloatArg)
    R = ("reference", StructureArg)
    reg = lambda name, desc, func: register(name, desc, func, logger=logger)

    reg("itopt target", CmdDesc(required=[("structure", StructureArg)],
                                synopsis="Set the structure whose chains IT-opt predicts"), itopt_target)
    reg("itopt distance", CmdDesc(
        required=[("atoms", AtomsArg)],
        keyword=[("with_", AtomsArg), ("target", FloatArg), ("tolerance", FloatArg), ("lower", FloatArg),
                 ("upper", FloatArg), W, R],
        synopsis="Distance restraint between two atoms (or two atom groups: 'with')"), itopt_distance)
    reg("itopt position", CmdDesc(
        required=[("atoms", AtomsArg)],
        keyword=[("frame", AtomsArg), ("to", AtomsArg), ("tolerance", FloatArg), W, ("use", SUBSET),
                 ("frame_use", SUBSET), R],
        required_arguments=["frame"],
        synopsis="Pin atoms to positions (current, 'to' markers/atoms, or reference) relative to frame atoms"),
        itopt_position)
    reg("itopt substructure", CmdDesc(
        required=[("atoms", AtomsArg)],
        keyword=[R, ("tolerance", FloatArg), W, ("use", SUBSET)],
        synopsis="Keep a substructure close (RMSD) to its current or reference conformation"),
        itopt_substructure)
    reg("itopt orientation", CmdDesc(
        required=[("domain1", AtomsArg)],
        keyword=[("with_", AtomsArg), R, ("angle_tol", FloatArg), ("dist_tol", FloatArg), W, ("use", SUBSET)],
        required_arguments=["with_"],
        synopsis="Restrain the relative orientation of two domains"), itopt_orientation)
    reg("itopt sketch", CmdDesc(optional=[("structure", StructureArg)],
                                synopsis="Make an editable copy of the target to drag domains/loops into place"),
        itopt_sketch)
    reg("itopt sketch restrain", CmdDesc(
        keyword=[("threshold", FloatArg), ("tolerance", FloatArg), ("rigid_tol", FloatArg),
                 ("min_domain", IntArg), ("angle_tol", FloatArg), W],
        synopsis="Turn segments moved in the sketch into restraints"), itopt_sketch_restrain)
    reg("itopt sketch close", CmdDesc(synopsis="Close the sketch and restore mouse modes"), itopt_sketch_close)
    reg("itopt ghosts", CmdDesc(optional=[("show", BoolArg)], synopsis="Show target-geometry ghost traces"),
        itopt_ghosts)
    reg("itopt mouse", CmdDesc(optional=[("mode", EnumOf(("distance", "sketch", "off")))],
                               keyword=[("button", EnumOf(("left", "middle", "right")))],
                               synopsis="Mouse mode: click two atoms -> distance restraint; sketch dragging"),
        itopt_mouse)
    reg("itopt list", CmdDesc(synopsis="List IT-opt restraints"), itopt_list)
    reg("itopt delete", CmdDesc(required=[("ids", RestOfLine)], synopsis="Delete restraints"), itopt_delete)
    reg("itopt save", CmdDesc(required=[("file_name", SaveFileNameArg)], synopsis="Save restraints (JSON)"),
        itopt_save)
    reg("itopt load", CmdDesc(required=[("file_name", OpenFileNameArg)], keyword=[("target", StructureArg)],
                              synopsis="Load restraints (JSON)"), itopt_load)
    reg("itopt run", CmdDesc(
        keyword=[("samples", IntArg), ("outer", IntArg), ("steps", IntArg), ("cycles", IntArg), ("lr", FloatArg),
                 ("msa", BoolArg), ("device", EnumOf(("mps", "cpu", "cuda"))), ("lambda_s", FloatArg),
                 ("lambda_z", FloatArg), ("baseline", BoolArg), ("guide_from", IntArg), ("seed", IntArg),
                 ("name", StringArg), ("restraint_scale", FloatArg)],
        synopsis="Run IT-opt of Protenix against the restraints"), itopt_run)
    reg("itopt stop", CmdDesc(synopsis="Stop the running IT-opt job"), itopt_stop)
    reg("itopt wait", CmdDesc(keyword=[("timeout", FloatArg)], synopsis="Block until the job finishes"), itopt_wait)
    reg("itopt status", CmdDesc(synopsis="Report IT-opt job status"), itopt_status)
    reg("itopt config", CmdDesc(keyword=[("python", StringArg), ("protenix", StringArg), ("runs_dir", StringArg)],
                                synopsis="Configure backend paths"), itopt_config)

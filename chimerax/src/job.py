"""Launch the IT-opt backend as a subprocess and follow its progress."""

import json
import os
import subprocess
import sys
import time

import numpy as np

from .state import get_state, RESULT_COLORS

BACKEND = os.path.join(os.path.dirname(__file__), "backend", "run_restraints.py")


def settings(session):
    from chimerax.core.settings import Settings

    class _S(Settings):
        AUTO_SAVE = {
            "python": os.path.expanduser("~/miniforge3/envs/itopt/bin/python"),
            "protenix": os.path.expanduser("~/Documents/code/ext/it_opt/protenix"),
            "runs_dir": os.path.expanduser("~/itopt_runs"),
        }

    s = getattr(session, "_itopt_settings", None)
    if s is None:
        s = _S(session, "ITopt")
        session._itopt_settings = s
    return s


class Job:
    def __init__(self, session, spec, out_dir):
        self.session = session
        self.spec = spec
        self.out_dir = out_dir
        self.status_path = os.path.join(out_dir, "status.json")
        self.seen_files = set()
        self.opened = {}  # prefix -> group model
        self.last_stage = None
        self.last_step_log = 0
        self.status = {}
        self.finished = False
        self.timer = None
        cs = settings(session)
        self.log = open(os.path.join(out_dir, "run.log"), "w")
        spec_path = os.path.join(out_dir, "spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=1)
        cmd = [cs.python, "-u", BACKEND, spec_path, "--protenix", cs.protenix]
        env = dict(os.environ)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        for k in ("PYTHONHOME", "PYTHONPATH"):  # don't leak ChimeraX's Python
            env.pop(k, None)
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                     cwd=out_dir, start_new_session=True)
        # key (chain, seqpos, name) -> (chain, author resnum, name) in result files
        self.numbering = {c["id"]: c["res_numbers"] for c in spec["chains"]}
        session.logger.info(f"IT-opt job started (pid {self.proc.pid}); output in {out_dir}")
        self._start_timer()

    # ----------------------------------------------------------- polling
    def _start_timer(self):
        if not self.session.ui.is_gui:
            return
        from Qt.QtCore import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(1500)

    def stop(self):
        if self.proc.poll() is None:
            import signal
            self.proc.send_signal(signal.SIGTERM)
            self.session.logger.info("IT-opt: stop requested; final samples will be drawn from the current embeddings")

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self._end()

    def _end(self):
        self.finished = True
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        try:
            self.log.close()
        except Exception:
            pass

    def poll(self):
        if self.finished:
            return
        logger = self.session.logger
        try:
            with open(self.status_path) as f:
                st = json.load(f)
        except (OSError, ValueError):
            st = None
        if st:
            self.status = st
            self._handle(st)
        rc = self.proc.poll()
        if rc is not None and (not st or st.get("stage") not in ("done", "error")):
            # process died without a final status
            tail = _tail(os.path.join(self.out_dir, "run.log"))
            logger.error(f"IT-opt backend exited with code {rc}.\n{tail}")
            self._end()
        elif st and st.get("stage") in ("done", "error") and rc is not None:
            self._end()
        state = get_state(self.session)
        state.changed()

    def _handle(self, st):
        logger = self.session.logger
        state = get_state(self.session)
        stage = st.get("stage")
        if stage != self.last_stage:
            self.last_stage = stage
            if stage == "error":
                logger.error(f"IT-opt failed: {st.get('error')}\n{st.get('traceback', '')}")
            elif stage == "done":
                pass
            else:
                logger.status(f"IT-opt: {stage}", log=True)
        if stage == "optimizing":
            msg = (f"IT-opt: outer {st['outer'] + 1}/{st['outers']}, step {st['step'] + 1}/{st['steps']}, "
                   f"restraint loss {st.get('loss') or 0:.3g}, ETA {_fmt_t(st.get('eta_s'))}")
            logger.status(msg)
            if st.get("restraints"):
                state.color_status(st["restraints"])
        # results
        new = [f for f in st.get("files", []) if f not in self.seen_files]
        if new:
            self.seen_files.update(new)
            groups = {}
            for f in new:
                prefix = os.path.basename(f).rsplit("_s", 1)[0]
                groups.setdefault(prefix, []).append(f)
            for prefix, files in groups.items():
                self._open_group(prefix, files)
        if stage == "done" and not getattr(self, "_reported", False):
            self._reported = True
            self._report(st)

    # ----------------------------------------------------- results display
    def _open_group(self, prefix, files):
        from chimerax.core.commands import run
        from chimerax.atomic import AtomicStructure
        session = self.session
        kind = "iter" if prefix.startswith("iter") else prefix
        if kind == "iter":  # keep only the latest guided iteration
            old = self.opened.pop("iter", None)
            if old is not None and not old.deleted:
                session.models.close([old])
        models = []
        for f in sorted(files):
            ms, _ = session.open_command.open_data(f, format="mmcif", log_info=False)
            models += [m for m in ms if isinstance(m, AtomicStructure)]
        if not models:
            return
        name = {"baseline": "ITopt baseline", "final": "ITopt final"}.get(kind, f"ITopt {prefix}")
        grp = session.models.add_group(models, name=name)
        self.opened[kind] = grp
        rms = [self._superpose(m) for m in models]
        color = RESULT_COLORS.get(kind, "orange")
        ids = ",".join(m.id_string for m in models)
        run(session, f"color #{ids} {color} target c; color #{ids} byhetero target a", log=False)
        if kind == "baseline":
            grp.display = False
        rms_txt = ", ".join(f"{r:.1f}" for r in rms if r is not None)
        session.logger.info(f"IT-opt: opened {name} (#{grp.id_string}); superposed on restraint frame atoms "
                            f"(RMSD {rms_txt} A)")

    def _superpose(self, model):
        from chimerax.geometry import align_points
        state = get_state(self.session)
        keys = state.alignment_keys()
        lut = {}
        for a in model.atoms:
            r = a.residue
            lut[(r.chain_id, r.number, a.name)] = a
        mov, fix = [], []
        for k in keys:
            nums = self.numbering.get(k[0])
            num = nums[k[1] - 1] if nums else k[1]
            a = lut.get((k[0], num, k[2]))
            t = state.atom_from_key(k)
            if a is not None and t is not None:
                mov.append(a.scene_coord)
                fix.append(t.scene_coord)
        if len(mov) < 3:
            return None
        tf, rms = align_points(np.array(mov), np.array(fix))
        model.scene_position = tf * model.scene_position
        return rms

    def _report(self, st):
        state = get_state(self.session)
        base = {x["id"]: x for x in st.get("baseline") or []}
        fin = {x["id"]: x for x in st.get("final") or []}
        state.color_status(st.get("final"), final=True)
        rows = []
        for r in state.restraints:
            b, f = base.get(r["id"]), fin.get(r["id"])
            unit = "deg" if r["type"] == "orientation" else "A"
            def cell(x):
                if not x:
                    return "-"
                return f"{x['value_mean']:.2f} {unit} ({sum(x['satisfied'])}/{len(x['satisfied'])} ok)"
            rows.append(f"<tr><td>{r['id']}</td><td>{r['type']}</td><td>{state.describe(r)}</td>"
                        f"<td>{cell(b)}</td><td>{cell(f)}</td></tr>")
        html = ("<b>IT-opt finished</b> in " + _fmt_t(st.get("t_total")) +
                f" &mdash; output: {self.out_dir}<br><table border=1 cellpadding=3>"
                "<tr><th>id</th><th>type</th><th>restraint</th><th>baseline</th><th>IT-opt final</th></tr>"
                + "".join(rows) + "</table>")
        self.session.logger.info(html, is_html=True)

    def wait(self, timeout=None):
        t0 = time.time()
        while not self.finished:
            self.poll()
            if self.finished:
                break
            if timeout is not None and time.time() - t0 > timeout:
                break
            time.sleep(1.0)


def _tail(path, n=25):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def _fmt_t(s):
    if s is None:
        return "?"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"

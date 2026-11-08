"""ITopt Restraints panel.  Every button issues a logged `itopt ...` command."""

from chimerax.core.tools import ToolInstance
from chimerax.core.commands import run

from .state import get_state


class ITOptTool(ToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False
    help = None

    def __init__(self, session, tool_name="ITopt Restraints"):
        super().__init__(session, tool_name)
        from chimerax.ui import MainToolWindow
        from Qt.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget,
                                  QSpinBox, QCheckBox, QDoubleSpinBox, QGroupBox, QGridLayout,
                                  QHeaderView, QProgressBar, QAbstractItemView)
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        lay = QVBoxLayout(parent)
        lay.setContentsMargins(4, 4, 4, 4)

        self.target_label = QLabel("Target: (none)")
        lay.addWidget(self.target_label)

        # ---- restraint creation
        box = QGroupBox("Add restraint from current selection")
        g = QGridLayout(box)

        def btn(text, tip, slot, r, c, span=1):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            g.addWidget(b, r, c, 1, span)
            return b

        self.tol = QDoubleSpinBox()
        self.tol.setRange(0.1, 50)
        self.tol.setValue(1.0)
        self.tol.setSuffix(" A tol")
        g.addWidget(self.tol, 0, 0)
        btn("Distance (2 atoms)", "Keep the distance between 2 selected atoms at its current value +/- tol",
            self._distance, 0, 1)
        btn("Substructure", "Keep the selected residues' CA RMSD to the current conformation <= tol",
            self._substructure, 0, 2)
        btn("Set frame := sel", "Remember the selection as the reference frame for 'Pin'", self._set_frame, 1, 0)
        btn("Pin sel in frame", "Pin selected atoms at their current positions relative to the frame",
            self._pin, 1, 1)
        self.frame_label = QLabel("frame: -")
        g.addWidget(self.frame_label, 1, 2)
        btn("Set domain 1 := sel", "Remember the selection as domain 1 for 'Orientation'", self._set_dom1, 2, 0)
        btn("Orientation dom1 vs sel", "Keep the current relative orientation of domain 1 and the selection",
            self._orient, 2, 1)
        self.dom_label = QLabel("domain 1: -")
        g.addWidget(self.dom_label, 2, 2)
        btn("Click-2-atoms distance", "Right-click two atoms (or cartoon residues) to add distance restraints",
            lambda: run(self.session, "itopt mouse distance"), 3, 0, 2)
        btn("Restore mouse", "Restore your mouse modes", lambda: run(self.session, "itopt mouse off"), 3, 2)
        lay.addWidget(box)

        # ---- sketch mode
        box = QGroupBox("Sketch: drag the structure into the shape you want")
        h = QHBoxLayout(box)
        for text, tip, cmd in [
            ("Make sketch copy", "Editable copy of the target; right-drag moves selected atoms, shift-right rotates",
             "itopt sketch"),
            ("Restrain moves", "Turn every moved segment of the sketch into position/orientation restraints",
             None),
            ("Close sketch", "Close the sketch and restore mouse modes", "itopt sketch close"),
        ]:
            b = QPushButton(text)
            b.setToolTip(tip)
            if cmd is None:
                b.clicked.connect(lambda: run(self.session,
                                              f"itopt sketch restrain tolerance {self.tol.value():g}"))
            else:
                b.clicked.connect(lambda _=False, c=cmd: self._sketch_cmd(c))
            h.addWidget(b)
        lay.addWidget(box)

        # ---- restraint table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["id", "type", "restraint", "status"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        b = QPushButton("Delete selected")
        b.clicked.connect(self._delete)
        row.addWidget(b)
        b = QPushButton("Delete all")
        b.clicked.connect(lambda: run(self.session, "itopt delete all"))
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        # ---- run settings
        box = QGroupBox("Run (Protenix IT-opt)")
        h = QHBoxLayout(box)

        def spin(label, lo, hi, val):
            h.addWidget(QLabel(label))
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            h.addWidget(s)
            return s

        self.samples = spin("samples", 1, 32, 4)
        self.outer = spin("outer", 1, 50, 5)
        self.steps = spin("steps", 10, 400, 100)
        self.msa = QCheckBox("MSA")
        self.msa.setChecked(True)
        h.addWidget(self.msa)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        h.addWidget(self.run_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(lambda: run(self.session, "itopt stop"))
        h.addWidget(self.stop_btn)
        lay.addWidget(box)
        self.progress = QProgressBar()
        self.progress.setFormat("idle")
        self.progress.setValue(0)
        lay.addWidget(self.progress)

        self.state = get_state(session)
        self.state.listeners.append(self.refresh)
        tw.manage(placement="side")
        self.refresh()

    def delete(self):
        try:
            self.state.listeners.remove(self.refresh)
        except ValueError:
            pass
        super().delete()

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, "ITopt Restraints")

    # ------------------------------------------------------------- actions
    def _sketch_cmd(self, cmd):
        if cmd == "itopt sketch" and self.state.target is None:
            from chimerax.atomic import selected_atoms, AtomicStructure
            structs = selected_atoms(self.session).unique_structures
            if len(structs) == 0:
                structs = [m for m in self.session.models if isinstance(m, AtomicStructure)]
            if len(structs) != 1:
                self.session.logger.warning("Select an atom of the structure to sketch (several are open)")
                return
            cmd += " " + structs[0].atomspec
        run(self.session, cmd)

    def _distance(self):
        run(self.session, f"itopt distance sel tolerance {self.tol.value():g}")

    def _substructure(self):
        run(self.session, f"itopt substructure sel tolerance {self.tol.value():g}")

    def _set_frame(self):
        from chimerax.atomic import selected_atoms
        a = selected_atoms(self.session)
        if len(a) == 0:
            self.session.logger.warning("Nothing selected")
            return
        self.state.frame_spec = _spec_for(self.session, a)
        self.frame_label.setText(f"frame: {self.state.frame_spec}")

    def _pin(self):
        spec = getattr(self.state, "frame_spec", None)
        if not spec:
            self.session.logger.warning("Set a frame first (select anchor residues, 'Set frame := sel')")
            return
        run(self.session, f"itopt position sel frame {spec} tolerance {self.tol.value():g}")

    def _set_dom1(self):
        from chimerax.atomic import selected_atoms
        a = selected_atoms(self.session)
        if len(a) == 0:
            self.session.logger.warning("Nothing selected")
            return
        self.state.dom1_spec = _spec_for(self.session, a)
        self.dom_label.setText(f"domain 1: {self.state.dom1_spec}")

    def _orient(self):
        spec = getattr(self.state, "dom1_spec", None)
        if not spec:
            self.session.logger.warning("Set domain 1 first")
            return
        run(self.session, f"itopt orientation {spec} with sel distTol {self.tol.value():g}")

    def _delete(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        ids = [self.table.item(r, 0).text() for r in rows]
        if ids:
            run(self.session, "itopt delete " + " ".join(ids))

    def _run(self):
        run(self.session, f"itopt run samples {self.samples.value()} outer {self.outer.value()} "
                          f"steps {self.steps.value()} msa {str(self.msa.isChecked()).lower()}")

    # -------------------------------------------------------------- refresh
    def refresh(self):
        from Qt.QtWidgets import QTableWidgetItem
        from Qt.QtGui import QColor
        st = self.state
        self.target_label.setText(f"Target: {st.target}" if st.target is not None else "Target: (none)")
        self.table.setRowCount(len(st.restraints))
        colors = {"pending": QColor(255, 220, 120), "ok": QColor(150, 230, 160), "bad": QColor(240, 150, 150)}
        for i, r in enumerate(st.restraints):
            status = r.get("_status", "pending")
            txt = status if "_value" not in r else f"{r['_value']:.2f} ({100 * r['_frac']:.0f}% ok)"
            for j, v in enumerate([r["id"], r["type"], st.describe(r), txt]):
                it = QTableWidgetItem(v)
                if j == 3:
                    it.setBackground(colors[status])
                self.table.setItem(i, j, it)
        job = st.job
        running = job is not None and not job.finished
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if job is not None:
            s = job.status
            if s.get("stage") == "optimizing":
                done = s["outer"] * s["steps"] + s["step"] + 1
                tot = s["outers"] * s["steps"]
                self.progress.setMaximum(tot)
                self.progress.setValue(done)
                self.progress.setFormat(f"outer {s['outer'] + 1}/{s['outers']}  step {s['step'] + 1}/{s['steps']}"
                                        f"  loss {s.get('loss') or 0:.3g}")
            elif s.get("stage"):
                self.progress.setFormat(s["stage"])
                if s["stage"] == "done":
                    self.progress.setValue(self.progress.maximum())


def _spec_for(session, atoms):
    from chimerax.atomic import concise_residue_spec
    return concise_residue_spec(session, atoms.unique_residues)

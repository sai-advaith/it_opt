"""Mouse modes: click two atoms -> distance restraint; sketch drag bindings."""

from chimerax.mouse_modes import MouseMode


class DistanceRestraintMouseMode(MouseMode):
    name = "itopt distance"

    def __init__(self, session):
        super().__init__(session)
        self._first = None
        self.tolerance = 1.0

    def mouse_down(self, event):
        x, y = event.position()
        pick = self.session.main_view.picked_object(x, y)
        atom = _picked_atom(pick)
        log = self.session.logger
        if atom is None:
            log.status("IT-opt distance: click an atom (or a residue's cartoon for its CA)")
            return
        if self._first is None or self._first.deleted:
            self._first = atom
            log.status(f"IT-opt distance: first atom {atom}; click the second atom")
            return
        a, self._first = self._first, None
        if a is atom:
            return
        from chimerax.core.commands import run
        run(self.session, f"itopt distance {a.string(style='command')} {atom.string(style='command')} "
                          f"tolerance {self.tolerance:g}")

    def mouse_drag(self, event):
        pass

    def mouse_up(self, event):
        pass


def _picked_atom(pick):
    if pick is None:
        return None
    a = getattr(pick, "atom", None)
    if a is not None:
        return a
    r = getattr(pick, "residue", None)
    if r is not None:
        return r.find_atom("CA") or (r.atoms[0] if len(r.atoms) else None)
    return None


def _modes(session):
    if not session.ui.is_gui:
        return None
    return session.ui.mouse_modes


def _save(session, buttons):
    from .state import get_state
    st = get_state(session)
    mm = _modes(session)
    if mm is None or st._saved_bindings is not None:
        return
    saved = []
    for b in mm.bindings:
        if b.button in buttons:
            saved.append((b.button, list(b.modifiers), b.mode))
    st._saved_bindings = (buttons, saved)


def restore_modes(session):
    from .state import get_state
    st = get_state(session)
    mm = _modes(session)
    if mm is None or st._saved_bindings is None:
        return
    buttons, saved = st._saved_bindings
    for b in list(mm.bindings):
        if b.button in buttons:
            mm.remove_binding(button=b.button, modifiers=b.modifiers)
    for button, mods, mode in saved:
        mm.bind_mouse_mode(mouse_button=button, mouse_modifiers=mods, mode=mode)
    st._saved_bindings = None
    session.logger.info("IT-opt: mouse modes restored")


def bind_sketch_modes(session):
    mm = _modes(session)
    if mm is None:
        return
    _save(session, ("right",))
    mm.bind_mouse_mode(mouse_button="right", mouse_modifiers=[], mode=mm.named_mode("translate selected atoms"))
    mm.bind_mouse_mode(mouse_button="right", mouse_modifiers=["shift"],
                       mode=mm.named_mode("rotate selected atoms"))
    session.logger.info("IT-opt: right-drag = translate selected atoms, shift-right-drag = rotate selected atoms "
                        "('itopt mouse off' restores your mouse modes)")


def bind_distance_mode(session, button="right"):
    mm = _modes(session)
    if mm is None:
        return
    mode = mm.named_mode(DistanceRestraintMouseMode.name)
    if mode is None:
        mode = DistanceRestraintMouseMode(session)
        mm.add_mode(mode)
    _save(session, (button,))
    mm.bind_mouse_mode(mouse_button=button, mouse_modifiers=[], mode=mode)
    session.logger.info(f"IT-opt: {button}-click two atoms (or cartoon residues -> CA) to add a distance "
                        "restraint at their current distance ('itopt mouse off' restores your mouse modes)")

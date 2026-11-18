"""Click an atom, get the handbook. The headline interaction of the tool.

Registered as a mouse mode rather than bound to plain selection, so it does
not fight ChimeraX's normal picking. Assign it from the Mouse Modes toolbar
or with:  mousemode right "hit picking handbook"
"""

from __future__ import annotations

from chimerax.mouse_modes import MouseMode

MODE_NAME = "hit picking handbook"


class HandbookMouseMode(MouseMode):
    name = MODE_NAME
    icon_file = None

    def __init__(self, session):
        super().__init__(session)

    def mouse_down(self, event):
        super().mouse_down(event)
        x, y = event.position()
        pick = self.session.main_view.picked_object(x, y)
        self._handle(pick)

    def _handle(self, pick):
        from .state import get_state
        from . import render

        session = self.session
        st = get_state(session)
        if st.handbook is None:
            session.logger.warning(
                "hitpick: handbook data not built; run tools/build_data.py")
            return
        if st.molsys is None:
            session.logger.warning(
                "hitpick: nothing analysed yet. Run: hitpick analyze")
            return

        atom = getattr(pick, "atom", None)
        if atom is None:
            # clicking an interaction line explains the interaction instead
            pb = getattr(pick, "pseudobond", None)
            if pb is not None:
                self._explain_pseudobond(st, pb)
            return

        idx = st.atom_index(atom)
        if idx is None:
            session.logger.info(
                "hitpick: %s is not part of the analysed system" % atom)
            return

        hit = st.handbook.for_atom(st.molsys, idx)
        st.last_hit = hit
        from .cmd import _log_hit
        _log_hit(session, hit)
        if hit.highlight:
            render.highlight(session, hit.highlight, st.all_atoms)
        st.notify()

    def _explain_pseudobond(self, st, pb):
        session = self.session
        result = st.result
        if result is None:
            return
        kind = getattr(pb, "hitpick_kind", None)
        if kind is None:
            return
        # find the interaction this line was drawn for
        best = None
        for it in result.interactions:
            if it.kind != kind:
                continue
            try:
                d = abs(it.distance - pb.length)
            except Exception:
                continue
            if best is None or d < best[0]:
                best = (d, it)
        if best is None:
            return
        hit = st.handbook.for_interaction(best[1])
        st.last_hit = hit
        from .cmd import _log_hit
        _log_hit(session, hit)
        st.notify()


def register_mouse_mode(session):
    if not hasattr(session, "ui") or not hasattr(session.ui, "mouse_modes"):
        return False          # nogui session
    mm = session.ui.mouse_modes
    try:
        if mm.named_mode(MODE_NAME) is not None:
            return True
    except Exception:
        pass
    mm.add_mode(HandbookMouseMode(session))
    return True

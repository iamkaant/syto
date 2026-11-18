"""Per-session state: the receptor context, the current pose, the toggles.

Kept in one place so the command layer, the mouse mode, the ViewDock link and
the HTML panel all see the same thing.
"""

from __future__ import annotations

from hitpicking.analysis import Options, ReceptorContext
from hitpicking.handbook import Handbook

_STATE = {}


class HitPickState:
    def __init__(self, session):
        self.session = session
        self.options = Options()
        self.receptor_model = None
        self.context = None          # ReceptorContext
        self.ligand_model = None
        self.result = None           # PoseResult
        self.molsys = None
        self.all_atoms = []          # index -> ChimeraX Atom
        self.handbook = Handbook.load_or_none()
        self.listeners = []          # callables notified when the pose changes
        self.last_hit = None         # last HandbookHit

    # ------------------------------------------------------------- helpers

    @property
    def ready(self):
        return self.context is not None

    def atom_index(self, atom):
        for i, a in enumerate(self.all_atoms):
            if a is atom:
                return i
        return None

    def notify(self):
        for cb in list(self.listeners):
            try:
                cb(self)
            except Exception as e:
                self.session.logger.warning("hitpick listener failed: %s" % e)

    def add_listener(self, cb):
        if cb not in self.listeners:
            self.listeners.append(cb)

    def remove_listener(self, cb):
        if cb in self.listeners:
            self.listeners.remove(cb)


def get_state(session, create=True):
    st = _STATE.get(session)
    if st is None and create:
        st = HitPickState(session)
        _STATE[session] = st
    return st


def clear_state(session):
    _STATE.pop(session, None)

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
        self.last_torsion = None     # Torsion for the last clicked bond
        self._sel_handler = None     # selection-changed trigger handler
        self._suppress_selection = False  # True while we set the selection
        self._atom_index_cache = None

    # ------------------------------------------------------------- helpers

    @property
    def ready(self):
        return self.context is not None

    def atom_index(self, atom):
        """Index of a ChimeraX Atom in the analysed system, or None.

        Cached: this runs on every selection change, and a linear scan over a
        few thousand atoms per click is wasteful. The cache is keyed on the
        identity of the atom list, so it rebuilds whenever a new pose is
        analysed.
        """
        cache = getattr(self, "_atom_index_cache", None)
        if cache is None or cache[0] is not self.all_atoms:
            cache = (self.all_atoms,
                     {id(a): i for i, a in enumerate(self.all_atoms)})
            self._atom_index_cache = cache
        return cache[1].get(id(atom))

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

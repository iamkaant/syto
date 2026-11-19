"""`hitpick learn` - active learning over the grades given at a party.

Nothing here is imported until one of these commands is run, so a user who
only wants the handbook never loads the model, the features or numpy's
linear algebra paths.

Grades come from ViewDock's own Rating column where it is running - good,
maybe and bad, with the g/m/b hotkeys - so there is no second place to
record an opinion. Without ViewDock, `hitpick learn grade` sets one
directly on the displayed pose.
"""

from __future__ import annotations

import os

_SESSIONS = {}


class LearnState:
    """The learner plus the pose features gathered so far."""

    def __init__(self, session):
        from hitpicking.al import Learner
        self.session = session
        self.learner = Learner(seed=0)
        self.keys = {}            # key -> ChimeraX structure
        self.last_ranked = []
        self.waiting = 0          # rated in ViewDock but not yet featurised

    # ------------------------------------------------------------ poses

    def key_for(self, structure):
        data = getattr(structure, "viewdock_data", None) or {}
        name = data.get("Name") or structure.name
        return "%s|%s" % (structure.id_string, str(name).strip())

    def add(self, structure, result):
        key = self.key_for(structure)
        self.learner.add_pose(key, result)
        self.keys[key] = structure
        return key

    # ----------------------------------------------------------- grades

    def pull_viewdock_ratings(self):
        """Take grades from ViewDock's Rating column.

        ViewDock stores 0 unrated, 1 bad, 2 maybe, 3 good, which is already
        the ordinal scale the model wants.

        Returns (applied, waiting) where `waiting` counts poses that are
        rated but not yet featurised. Those used to be dropped in silence,
        so grading in ViewDock before pressing "Analyse poses" lost the lot.
        """
        from . import viewdock_link as vl
        n = waiting = 0
        for structure, rating in vl.rated_poses(self.session):
            key = self.key_for(structure)
            if key not in self.learner.features:
                waiting += 1
                continue
            if self.learner.grades.rank(key) != int(rating) - 1:
                self.learner.grade(key, int(rating) - 1)
            n += 1
        self.waiting = waiting
        return n, waiting


def state(session, create=True):
    st = _SESSIONS.get(session)
    if st is None and create:
        st = LearnState(session)
        _SESSIONS[session] = st
    return st


def clear(session):
    _SESSIONS.pop(session, None)


# ------------------------------------------------------------- analysing

def collect(session, limit=None, quiet=False):
    """Analyse every pose ViewDock knows about and featurise it.

    This is the slow step - a few hundredths of a second per pose - and it
    has to happen before anything can be learned, so it is explicit rather
    than hidden inside `train`.
    """
    from .state import get_state
    from . import viewdock_link as vl
    from .cmd import hitpick_analyze

    hp = get_state(session)
    if not hp.ready:
        from .cmd import hitpick_setup
        hitpick_setup(session)
        hp = get_state(session)

    st = state(session)
    # Rated poses first. Analysing is the slow step and a real session can
    # hold a few thousand poses, so putting the graded ones at the front
    # means the model becomes trainable in the first seconds rather than
    # after the whole set has been ground through.
    scored = vl.iter_rated(session)
    structures = ([s for s, r in scored if r]
                  + [s for s, r in scored if not r])
    if not structures:
        from . import adapter
        structures = adapter.find_ligand_structures(
            session, exclude=[hp.receptor_model] if hp.receptor_model else [])
    if limit:
        structures = structures[:limit]
    if not structures:
        return 0

    done = 0
    for s in structures:
        if s is hp.receptor_model:
            continue
        key = st.key_for(s)
        if key in st.learner.features:
            continue
        try:
            result = hitpick_analyze(session, s)
        except Exception:
            continue
        # hitpick_analyze returns a summary; the full result is on the state
        if hp.result is not None:
            st.add(s, hp.result)
            done += 1
        if not quiet and done and done % 50 == 0:
            session.logger.status("hitpick learn: analysed %d poses" % done)
    return done


# ------------------------------------------------------------- reporting

def describe(session, pull=True):
    """Current state, taking in any ViewDock ratings first.

    Pulling here rather than only at training time is what lets the panel
    show a grade the moment it is given in ViewDock.
    """
    st = state(session)
    if pull:
        try:
            st.pull_viewdock_ratings()
        except Exception:
            pass
    lr = st.learner
    counts = lr.grades.counts()
    ok, why = lr.can_train()
    return {
        "poses": len(lr.features),
        "graded": lr.n_graded,
        "grades": counts,
        "trained_on": lr.trained_on,
        "ready": bool(lr.committee is not None),
        "can_train": ok,
        "why_not": why,
        "feature_version": lr.feature_version,
        "waiting": st.waiting,
    }


def save(session, path):
    import json
    st = state(session)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(st.learner.state(), fh)
    return path


def load(session, path):
    import json
    from hitpicking.al import Learner
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    st = state(session)
    st.learner = Learner.from_state(data)
    return st.learner

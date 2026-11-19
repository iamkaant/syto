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

def _features_for(hp, lig, job):
    """Analyse one pose for learning only - no drawing, no table update.

    The display path (`hitpick analyze`) also redraws pseudobonds, relabels
    and notifies the panel. Doing that for every pose in a bulk collect is
    both wasted and, in a window, far more expensive than the analysis
    itself, so this is a separate and much leaner route.

    Only the first step touches ChimeraX, so only the first step goes back
    to the GUI thread.
    """
    from hitpicking.analysis import PoseResult
    from hitpicking.nci.detect import detect_interactions
    from hitpicking.perceive.features import perceive_all
    from . import adapter

    def extract():
        ms, _atoms = adapter.from_structures(
            [hp.receptor_model, lig], ligand_structures=[lig])
        # read the scalars here too; `lig` must not be touched off-thread
        return ms, str(lig.name), int(hp.receptor_model.num_atoms)

    ms, name, n_rec = job.on_main(extract)

    job.checkpoint()
    feats = perceive_all(ms)
    res = detect_interactions(
        ms, enabled=hp.options.enabled_nci,
        include_weak_hbonds=hp.options.include_weak_hbonds,
        features=feats)
    job.checkpoint()
    strain = None
    if hp.options.detect_strain:
        from hitpicking.strain import analyze_strain
        strain = analyze_strain(ms)
    return PoseResult(name, res.interactions, res.warnings, res.counts, {},
                      combined=ms, n_receptor=n_rec, strain=strain)


def _store(st, key, structure, vector):
    """Record one featurised pose. Always called on the GUI thread."""
    st.learner.add_features(key, vector)
    st.keys[key] = structure


def collect(session, limit=None, quiet=False, job=None):
    """Analyse every pose ViewDock knows about and featurise it.

    This is the slow step - about 0.14 s a pose - so with a few thousand
    loaded it is minutes of work and belongs on a background thread. Pass a
    `job` to get progress and cancellation; without one it runs inline, as
    it does under test.
    """
    from .state import get_state
    from . import viewdock_link as vl
    from . import background

    hp = get_state(session)
    if not hp.ready:
        from .cmd import hitpick_setup
        hitpick_setup(session)
        hp = get_state(session)

    st = state(session)
    if job is None:
        job = background.Job(session, "analysing")

    # Rated poses first. Analysing is the slow step and a real session can
    # hold a few thousand poses, so putting the graded ones at the front
    # means the model becomes trainable in the first seconds rather than
    # after the whole set has been ground through.
    scored = job.on_main(vl.iter_rated, session)
    structures = ([s for s, r in scored if r]
                  + [s for s, r in scored if not r])
    if not structures:
        from . import adapter
        structures = job.on_main(
            adapter.find_ligand_structures, session,
            exclude=[hp.receptor_model] if hp.receptor_model else [])
    if limit:
        structures = structures[:limit]
    if not structures:
        return 0

    todo = []
    for s in structures:
        if s is hp.receptor_model:
            continue
        key = st.key_for(s)
        if key in st.learner.features:
            continue
        todo.append((s, key))

    done = 0
    total = len(todo)
    for s, key in todo:
        job.checkpoint()
        try:
            result = _features_for(hp, s, job)
        except background.Cancelled:
            raise
        except Exception:
            continue
        # Featurise on this thread - it is numpy over the detached
        # MolSystem - but write into the learner on the GUI thread. The
        # panel reads those same dictionaries whenever it refreshes, and a
        # dict being written from one thread while another walks it raises
        # "changed size during iteration" at random.
        from hitpicking.al.features import pose_features
        vector = pose_features(result)
        job.on_main(_store, st, key, s, vector)
        done += 1
        if done % 5 == 0 or done == total:
            job.report(done, total, "analysed %d of %d poses" % (done, total))
            if not quiet:
                session.logger.status(
                    "hitpick learn: analysed %d of %d poses" % (done, total))
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

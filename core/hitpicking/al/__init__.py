"""Active learning over the grades a party gives.

The idea, from AutoParty (Keiser lab): people grade a handful of poses, a
model learns from those grades, and the remaining poses are re-ordered so
the next ones shown are the ones worth looking at. With a few hundred
compounds and a room full of opinions, that is the difference between
seeing the interesting 5% and working alphabetically.

What this adds over a plain ranking is UNCERTAINTY. A committee of small
networks disagrees most where it has least evidence, and those are the
poses whose grades teach it the most - so "what should we look at next"
has a defensible answer rather than being the model's top prediction,
which would only confirm what it already believes.

Nothing here imports torch, ChimeraX or RDKit; it runs in the shared core
so the web front end can use it unchanged.
"""

from __future__ import annotations

import numpy as np

from .model import Committee, ordinal_targets, decode_ordinal
from .features import pose_features, feature_info, TOTAL_BITS
from .. import FEATURE_VERSION

__all__ = ["Learner", "Grades", "UNCERTAINTY", "DISAGREEMENT", "PREDICTION",
           "pose_features", "feature_info"]

# Acquisition strategies: how to choose what to show next.
UNCERTAINTY = "uncertainty"     # where the committee disagrees with itself
DISAGREEMENT = "disagreement"   # where it disagrees with the human grade
PREDICTION = "prediction"       # simply the best predicted, for review

DEFAULT_GRADES = ("bad", "maybe", "good")


class Grades:
    """Human grades, keyed however the caller identifies a pose."""

    def __init__(self, options=DEFAULT_GRADES):
        self.options = tuple(options)
        self._by_key = {}

    def set(self, key, grade):
        if grade is None:
            self._by_key.pop(key, None)
            return
        if isinstance(grade, str):
            if grade not in self.options:
                raise ValueError("unknown grade %r; expected one of %s"
                                 % (grade, ", ".join(self.options)))
            grade = self.options.index(grade)
        self._by_key[key] = int(grade)

    def get(self, key):
        return self._by_key.get(key)

    def rank(self, key):
        return self._by_key.get(key)

    def label(self, key):
        r = self._by_key.get(key)
        return None if r is None else self.options[r]

    def __len__(self):
        return len(self._by_key)

    def __contains__(self, key):
        return key in self._by_key

    def keys(self):
        return list(self._by_key)

    def counts(self):
        out = {g: 0 for g in self.options}
        for r in self._by_key.values():
            out[self.options[r]] += 1
        return out


class Learner:
    """Holds the features, the grades and the committee."""

    def __init__(self, options=DEFAULT_GRADES, committee_size=3,
                 n_neurons=256, hidden_layers=2, dropout=0.2, lr=1e-3,
                 weight_decay=1e-2, seed=0):
        self.options = tuple(options)
        self.grades = Grades(self.options)
        self.features = {}          # key -> vector
        self.committee = None
        self.feature_version = FEATURE_VERSION
        self.config = {"committee_size": committee_size,
                       "n_neurons": n_neurons, "hidden_layers": hidden_layers,
                       "dropout": dropout, "lr": lr,
                       "weight_decay": weight_decay, "seed": seed}
        self.trained_on = 0
        self.history = []
        # Per-column standardisation, fitted on the graded set. Without it
        # the ligand fingerprint block carries roughly eighteen times the
        # total magnitude of the descriptor block, so the clash count and
        # the interaction counts - the features a grader actually reasons
        # about - are swamped before the first layer sees them.
        self._mean = None
        self._scale = None

    # ------------------------------------------------------------ data

    def add_pose(self, key, result):
        """Record the feature vector for an analysed pose."""
        self.features[key] = pose_features(result)
        return self.features[key]

    def add_features(self, key, vector):
        v = np.asarray(vector, dtype=np.float64)
        if v.shape != (TOTAL_BITS,):
            raise ValueError("feature vector is %s, expected (%d,)"
                             % (v.shape, TOTAL_BITS))
        self.features[key] = v

    def grade(self, key, grade):
        self.grades.set(key, grade)

    @property
    def n_graded(self):
        return sum(1 for k in self.grades.keys() if k in self.features)

    # --------------------------------------------------------- training

    def can_train(self, minimum=8):
        """Enough grades, and more than one distinct grade among them."""
        keys = [k for k in self.grades.keys() if k in self.features]
        if len(keys) < minimum:
            return False, ("%d graded pose(s); at least %d are needed"
                           % (len(keys), minimum))
        ranks = {self.grades.rank(k) for k in keys}
        if len(ranks) < 2:
            only = self.options[next(iter(ranks))]
            return False, ("every graded pose is '%s'; the model needs at "
                           "least two different grades to learn anything"
                           % only)
        return True, ""

    def _fit_scaler(self, X):
        self._mean = X.mean(axis=0)
        sd = X.std(axis=0)
        # a constant column carries no information; leaving its scale at 1
        # keeps it at zero after centring rather than exploding it
        sd[sd < 1e-8] = 1.0
        self._scale = sd

    def _apply_scaler(self, X):
        if self._mean is None:
            return X
        return (X - self._mean) / self._scale

    def train(self, epochs=100, batch_size=16, patience=20, minimum=8):
        ok, why = self.can_train(minimum)
        if not ok:
            raise ValueError(why)
        keys = [k for k in self.grades.keys() if k in self.features]
        X = np.stack([self.features[k] for k in keys])
        self._fit_scaler(X)
        X = self._apply_scaler(X)
        Y = ordinal_targets([self.grades.rank(k) for k in keys],
                            len(self.options))
        cfg = dict(self.config)
        size = cfg.pop("committee_size")
        seed = cfg.pop("seed")
        self.committee = Committee(X.shape[1], len(self.options), size=size,
                                   seed=seed, **cfg)
        self.committee.fit(X, Y, epochs=epochs, batch_size=batch_size,
                           patience=patience)
        self.trained_on = len(keys)
        self.history = self.committee.history
        return self

    # ------------------------------------------------------- prediction

    def predict(self, keys=None):
        """{key: (grade, rank, uncertainty, error_or_None)}."""
        if self.committee is None:
            raise ValueError("the model has not been trained yet")
        keys = list(keys if keys is not None else self.features)
        keys = [k for k in keys if k in self.features]
        if not keys:
            return {}
        X = self._apply_scaler(np.stack([self.features[k] for k in keys]))
        mean, var = self.committee.predict(X)
        ranks = decode_ordinal(mean)
        unc = var.mean(axis=1)
        out = {}
        for k, r, u in zip(keys, ranks, unc):
            human = self.grades.rank(k)
            err = None if human is None else abs(int(r) - int(human))
            out[k] = {"grade": self.options[int(r)], "rank": int(r),
                      "uncertainty": float(u), "error": err}
        return out

    def rank_poses(self, keys=None, strategy=UNCERTAINTY,
                   exclude_graded=True):
        """Poses in the order they are worth looking at next."""
        preds = self.predict(keys)
        rows = []
        for k, p in preds.items():
            if exclude_graded and k in self.grades:
                continue
            rows.append((k, p))
        if strategy == UNCERTAINTY:
            rows.sort(key=lambda kp: -kp[1]["uncertainty"])
        elif strategy == PREDICTION:
            rows.sort(key=lambda kp: (-kp[1]["rank"], -kp[1]["uncertainty"]))
        elif strategy == DISAGREEMENT:
            # only meaningful for poses that HAVE a grade, so the usual
            # exclude_graded must be turned off by the caller
            rows = [(k, p) for k, p in preds.items() if p["error"] is not None]
            rows.sort(key=lambda kp: (-(kp[1]["error"] or 0),
                                      -kp[1]["uncertainty"]))
        else:
            raise ValueError("unknown strategy %r" % strategy)
        return rows

    # ------------------------------------------------------ persistence

    def state(self):
        return {
            "feature_version": self.feature_version,
            "options": list(self.options),
            "config": self.config,
            "trained_on": self.trained_on,
            "mean": None if self._mean is None else self._mean.tolist(),
            "scale": None if self._scale is None else self._scale.tolist(),
            "grades": {str(k): v for k, v in self.grades._by_key.items()},
            "committee": None if self.committee is None
            else self.committee.state(),
        }

    @classmethod
    def from_state(cls, st):
        if st.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                "this model was trained with feature version %s but the code "
                "now produces version %s; the columns no longer mean the same "
                "thing, so it cannot be loaded. Retrain from the grades."
                % (st.get("feature_version"), FEATURE_VERSION))
        cfg = dict(st.get("config") or {})
        lr = cls(options=tuple(st.get("options") or DEFAULT_GRADES), **cfg)
        for k, v in (st.get("grades") or {}).items():
            lr.grades.set(k, int(v))
        if st.get("mean") is not None:
            lr._mean = np.asarray(st["mean"], dtype=np.float64)
            lr._scale = np.asarray(st["scale"], dtype=np.float64)
        if st.get("committee"):
            lr.committee = Committee.from_state(st["committee"])
        lr.trained_on = st.get("trained_on", 0)
        return lr

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
from .features import (pose_features, feature_info, score_block, TOTAL_BITS,
                       BLOCKS, SCORE_SLICE)
from . import scoreterms
from .scoreterms import DEFAULT_TERMS
from .. import FEATURE_VERSION

__all__ = ["Learner", "Grades", "UNCERTAINTY", "DISAGREEMENT", "PREDICTION",
           "pose_features", "feature_info", "scoreterms"]

# Acquisition strategies: how to choose what to show next.
UNCERTAINTY = "uncertainty"     # where the committee disagrees with itself
DISAGREEMENT = "disagreement"   # where it disagrees with the human grade
PREDICTION = "prediction"       # simply the best predicted, for review

DEFAULT_GRADES = ("bad", "maybe", "good")


def _numeric(scores):
    """Only the plain numbers; names, SMILES and flags are not features."""
    out = {}
    for k, v in (scores or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out[str(k)] = float(v)
    return out


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
                 weight_decay=1e-2, seed=0, score_terms=None):
        self.options = tuple(options)
        self.grades = Grades(self.options)
        self.features = {}          # key -> vector
        # The pose's own score terms, kept beside the vector so the
        # selection can be changed later without re-analysing: the score
        # slots are rewritten from these. Cheap - a handful of floats a
        # pose against a 2069-wide vector.
        self.scores = {}            # key -> {term: value}
        self.committee = None
        self.feature_version = FEATURE_VERSION
        self.score_terms = scoreterms.normalize(
            DEFAULT_TERMS if score_terms is None else score_terms)
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
        self.features[key] = pose_features(result, terms=self.score_terms)
        self.scores[key] = _numeric(getattr(result, "scores", None))
        return self.features[key]

    def add_features(self, key, vector, scores=None):
        v = np.asarray(vector, dtype=np.float64)
        if v.shape != (TOTAL_BITS,):
            raise ValueError("feature vector is %s, expected (%d,)"
                             % (v.shape, TOTAL_BITS))
        self.features[key] = v
        if scores is not None:
            self.scores[key] = _numeric(scores)

    # ------------------------------------------------- score term choice

    def set_score_terms(self, terms):
        """Change which of the pose's own numbers the model may learn from.

        The score slots are rewritten in place from the dictionaries kept
        by `add_pose`, so a few thousand poses do not have to be analysed
        again - which on a real set is minutes. Everything outside those
        slots is untouched.

        A trained committee is discarded: its first layer was fitted to
        columns that now hold different quantities, and keeping it would
        make every prediction quietly wrong. The grades survive, so
        retraining is one call.
        """
        terms = scoreterms.normalize(terms)
        if terms == self.score_terms:
            return 0
        self.score_terms = terms
        n = 0
        for key, vec in self.features.items():
            vec[SCORE_SLICE] = score_block(self.scores.get(key), terms)
            n += 1
        self.committee = None
        self.history = []
        self.trained_on = 0
        self._mean = self._scale = None
        return n

    def observed_scores(self):
        """The score dictionaries of every featurised pose, for `suggest`."""
        return list(self.scores.values())

    def suggest_score_terms(self):
        return scoreterms.suggest(self.observed_scores())

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
        live = sd > 1e-8
        # a constant column carries no information; leaving its scale at 1
        # keeps it at zero after centring rather than exploding it
        sd[~live] = 1.0

        # Then weight whole blocks against one another. Per-column
        # standardisation alone gave the ligand fingerprint about 90% of
        # the variance reaching the first layer - not because it matters
        # more, but because it has around 765 varying columns against the
        # interaction block's 67. The model was a ligand model with
        # interaction trimmings, and two poses of one compound, which
        # differ only outside the ligand block, were nearly identical to
        # it.
        #
        # Each block is scaled to carry the same total variance. The
        # weights are folded into _scale, so predict(), state() and
        # from_state() need no changes and an old model stays readable.
        counts = [(lo, hi, int(live[lo:hi].sum())) for _, lo, hi in BLOCKS]
        present = [n for _, _, n in counts if n]
        if present:
            # keep the overall magnitude where it was, so the learning
            # rate still means what it meant
            total = float(sum(present))
            share = total / len(present)
            for lo, hi, n in counts:
                if n:
                    sd[lo:hi] = sd[lo:hi] * np.sqrt(n / share)
        self._scale = sd

    def block_variance(self, X=None):
        """Share of the first layer's input variance each block carries.

        Reported rather than assumed: the imbalance this corrects was
        invisible until it was measured.
        """
        keys = [k for k in self.grades.keys() if k in self.features]
        if X is None:
            if not keys:
                return {}
            X = np.stack([self.features[k] for k in keys])
        Z = self._apply_scaler(X)
        energy = {name: float((Z[:, lo:hi] ** 2).sum())
                  for name, lo, hi in BLOCKS}
        total = sum(energy.values()) or 1.0
        return {k: v / total for k, v in energy.items()}

    def _apply_scaler(self, X):
        if self._mean is None:
            return X
        return (X - self._mean) / self._scale

    # 128, not AutoParty's 16. Measured on two real hit-picking sessions
    # (1017 and 675 human grades, 5 stratified held-out splits each): the
    # held-out AUC is 0.821 vs 0.820 and 0.745 vs 0.742, which is a wash,
    # while a fit takes 3.4 s instead of 14.7 and 2.2 s instead of 10.1.
    # On this hardware the step is a 2069-wide matmul either way; at 16 it
    # is all Python loop overhead and no arithmetic.
    DEFAULT_BATCH = 128

    def train(self, epochs=100, batch_size=DEFAULT_BATCH, patience=20,
              minimum=8):
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
        """{key: {grade, rank, score, uncertainty, error_or_None}}.

        `rank` is the decoded grade, 0..n-1. `score` is the thermometer
        output before decoding, on the same 0..n-1 axis but continuous.

        Both are reported because they are for different jobs. `rank` is
        what to show a person: "good". `score` is what to SORT by. On a real
        list the decoded rank collapses thousands of poses into two or three
        tied blocks - 1472 unrated poses came out as 308 at rank 1 and 1164
        at rank 0, in no order at all within either - and a column like that
        cannot be picked down. The score separates the same poses into over
        1300 distinct values, which is the ordering the party wants.
        """
        if self.committee is None:
            raise ValueError("the model has not been trained yet")
        keys = list(keys if keys is not None else self.features)
        keys = [k for k in keys if k in self.features]
        if not keys:
            return {}
        X = self._apply_scaler(np.stack([self.features[k] for k in keys]))
        mean, var = self.committee.predict(X)
        ranks = decode_ordinal(mean)
        # Sum of the thermometer probabilities, less the always-on first
        # unit: the expected number of thresholds passed, which is the grade
        # axis without the rounding.
        raw = np.asarray(mean, dtype=float).sum(axis=1) - 1.0
        # decode_ordinal resolves a non-monotone row (0.9, 0.3, 0.8) at the
        # FIRST crossing, which the sum does not know about, so the two can
        # in principle disagree. On real data they never did - 0 rows out of
        # ~600 held-out poses - but a table showing "bad" beside 1.48 would
        # be indefensible, so the number is held inside the word's bucket.
        lo = len(self.options) - 1.0
        scores = np.clip(np.clip(raw, ranks - 0.5, ranks + 0.5), 0.0, lo)
        unc = var.mean(axis=1)
        out = {}
        for k, r, sc, u in zip(keys, ranks, scores, unc):
            human = self.grades.rank(k)
            err = None if human is None else abs(int(r) - int(human))
            out[k] = {"grade": self.options[int(r)], "rank": int(r),
                      "score": float(sc),
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
            "score_terms": list(self.score_terms),
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
        # Models saved before the selection existed were trained on DOCK's
        # seven terms, which is what DEFAULT_TERMS still is - so an old
        # file loads unchanged rather than being refused.
        terms = st.get("score_terms")
        lr = cls(options=tuple(st.get("options") or DEFAULT_GRADES),
                 score_terms=DEFAULT_TERMS if terms is None else terms, **cfg)
        for k, v in (st.get("grades") or {}).items():
            lr.grades.set(k, int(v))
        if st.get("mean") is not None:
            lr._mean = np.asarray(st["mean"], dtype=np.float64)
            lr._scale = np.asarray(st["scale"], dtype=np.float64)
        if st.get("committee"):
            lr.committee = Committee.from_state(st["committee"])
        lr.trained_on = st.get("trained_on", 0)
        return lr

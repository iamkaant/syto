"""What goes into the active-learning model, and in what proportion.

Two things here were measured rather than assumed, and both are easy to
regress silently because nothing crashes when they are wrong:

  - the three blocks must reach the first layer with comparable weight.
    Standardising per column is not enough: the ligand fingerprint has
    roughly 765 varying columns to the interaction block's 67, which gave
    it ~90% of the variance and made an "interaction model" into a ligand
    model.
  - the docking program's own score terms must actually arrive. They were
    reserved slots that sat at zero for every pose, because PoseResult was
    built with no scores at all.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hitpicking import FEATURE_VERSION  # noqa: E402
from hitpicking.al import Learner  # noqa: E402
from hitpicking.al.features import (BLOCKS, DOCK_TERMS, IFP_BITS,  # noqa: E402
                                    LIGAND_BITS, N_DESCRIPTORS, TOTAL_BITS,
                                    descriptors)


class FakeStrain:
    counts = {"clashes": 1, "internal_clashes": 0, "flagged_torsions": 2}
    worst_overlap = 0.3


class FakeResult:
    """Enough of a PoseResult for the descriptor block."""

    def __init__(self, scores=None, ms=None):
        self.scores = scores or {}
        self.combined = ms
        self.interactions = []
        self.counts = {"hbond": 2}
        self.strain = FakeStrain()


class FakeMS:
    def __init__(self, n=20):
        self.is_ligand = np.ones(n, dtype=bool)
        self.element = np.array(["C"] * n)
        self.formal_charge = np.zeros(n)
        self.aromatic = np.zeros(n, dtype=bool)
        self.meta = {"has_hydrogens": True}


def test_block_layout_adds_up():
    assert TOTAL_BITS == IFP_BITS + LIGAND_BITS + N_DESCRIPTORS
    assert BLOCKS[-1][2] == TOTAL_BITS
    # contiguous and non-overlapping
    for (_, _, end), (_, start, _) in zip(BLOCKS, BLOCKS[1:]):
        assert end == start


def test_dock_terms_reach_the_vector():
    scores = {"Total Energy": -53.1, "Van der Waals": -32.2,
              "Electrostatic": -21.3, "Ligand Polar Desolv": 6.3,
              "Ligand Apolar Desolv": -5.9, "Total Strain": 3.3,
              "Max Strain": 1.35}
    d = descriptors(FakeResult(scores, FakeMS()))
    for k, name in enumerate(DOCK_TERMS):
        assert d[14 + k] != 0.0, "%s did not reach the vector" % name
    assert d[14] == pytest.approx(-53.1 / 50.0)


def test_dock_sentinels_are_not_let_through():
    """DOCK writes 999.999 for a term it could not compute.

    Left alone that one value sets the column mean and standard deviation
    for the whole set, and every real pose standardises to roughly the
    same number.
    """
    d = descriptors(FakeResult({"Total Energy": 999.999}, FakeMS()))
    assert d[14] == 0.0
    d = descriptors(FakeResult({"Total Energy": float("nan")}, FakeMS()))
    assert d[14] == 0.0
    # a real, large-but-plausible energy still gets through
    d = descriptors(FakeResult({"Total Energy": -180.0}, FakeMS()))
    assert d[14] == pytest.approx(-3.6)


def test_missing_scores_are_harmless():
    d = descriptors(FakeResult({}, FakeMS()))
    assert d.shape == (N_DESCRIPTORS,)
    assert not np.isnan(d).any()
    assert all(d[14 + k] == 0.0 for k in range(len(DOCK_TERMS)))


def _learner_with(n=60, seed=0):
    """A learner whose blocks have very different column counts."""
    rng = np.random.default_rng(seed)
    lr = Learner(seed=0)
    for i in range(n):
        v = np.zeros(TOTAL_BITS)
        # interactions: few active columns, as in real data
        v[rng.integers(0, 40)] = rng.random()
        # ligand: many
        v[IFP_BITS + rng.integers(0, 700, 50)] = 1.0
        # descriptors: all of them, varying
        v[IFP_BITS + LIGAND_BITS:] = rng.random(N_DESCRIPTORS)
        lr.add_features("p%d" % i, v)
        lr.grade("p%d" % i, int(rng.integers(0, 3)))
    return lr


def test_blocks_are_balanced_after_scaling():
    """The point of the exercise: no block dominates by column count."""
    lr = _learner_with()
    X = np.stack([lr.features[k] for k in lr.features])
    lr._fit_scaler(X)
    share = lr.block_variance(X)
    assert set(share) == {"interactions", "ligand", "descriptors"}
    for name, v in share.items():
        assert 0.30 < v < 0.37, "%s carries %.1f%% of the variance" % (
            name, 100 * v)


def test_without_balancing_the_ligand_block_would_dominate():
    """Shows the problem is real, not theoretical."""
    lr = _learner_with()
    X = np.stack([lr.features[k] for k in lr.features])
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (X - mean) / sd                      # per-column only, the old way
    energy = {n: float((Z[:, lo:hi] ** 2).sum()) for n, lo, hi in BLOCKS}
    total = sum(energy.values())
    assert energy["ligand"] / total > 0.7, (
        "per-column standardisation alone should leave the ligand block "
        "dominant; if it no longer does, this test's fixture has drifted")


def test_balancing_does_not_wreck_training():
    lr = _learner_with(n=40)
    lr.train(epochs=20)
    preds = lr.predict()
    assert len(preds) == 40
    assert all(np.isfinite(p["uncertainty"]) for p in preds.values())


def test_a_version_1_model_is_refused():
    """The columns mean different things now; loading one silently would
    give confident nonsense."""
    lr = _learner_with(n=12)
    lr.train(epochs=5)
    st = lr.state()
    assert st["feature_version"] == FEATURE_VERSION == 2
    st["feature_version"] = 1
    with pytest.raises(ValueError, match="feature version"):
        Learner.from_state(st)


def test_scaler_survives_a_save_and_load():
    """The block weights are folded into _scale, so they must come back."""
    lr = _learner_with(n=20)
    lr.train(epochs=5)
    before = lr.predict()
    back = Learner.from_state(lr.state())
    for k, v in lr.features.items():
        back.add_features(k, v)
    after = back.predict()
    for k in before:
        assert before[k]["rank"] == after[k]["rank"]
        assert before[k]["uncertainty"] == pytest.approx(
            after[k]["uncertainty"], rel=1e-9)


# ---------------------------------------------------------------------------
# What you sort the table by.
#
# The decoded grade is three values. On a real list - 1472 unrated poses from
# a PLA2 run - it produced two blocks, 308 and 1164, with no order inside
# either, which is not something a party can pick down. The continuous score
# separated the same poses into more than 1300 distinct values.
# ---------------------------------------------------------------------------


def _toy_learner(n=90, seed=0):
    rng = np.random.default_rng(seed)
    lr = Learner(seed=seed)
    for i in range(n):
        v = rng.normal(size=TOTAL_BITS).astype(np.float32)
        rank = i % 3
        v[:64] += rank                      # a signal to find
        lr.add_features("p%d" % i, v)
        if i % 3 != 2 or i < n // 2:        # leave some ungraded
            lr.grade("p%d" % i, rank)
    lr.train(epochs=30, batch_size=16)
    return lr


def test_prediction_carries_a_continuous_score_as_well_as_a_grade():
    lr = _toy_learner()
    preds = lr.predict()
    assert preds
    for p in preds.values():
        assert p["rank"] in (0, 1, 2)
        assert isinstance(p["score"], float)
        assert 0.0 <= p["score"] <= 2.0
        # the number must never contradict the word beside it
        assert abs(p["score"] - p["rank"]) <= 0.5, p


def test_the_score_breaks_the_ties_the_decoded_rank_leaves():
    """The decoded grade cannot order a list; the score has to."""
    lr = _toy_learner()
    preds = lr.predict()
    ranks = {p["rank"] for p in preds.values()}
    scores = {round(p["score"], 6) for p in preds.values()}
    assert len(ranks) <= 3
    assert len(scores) > 3 * len(ranks), (len(scores), len(ranks))
    # and sorting by score agrees with the grade it decodes to
    rows = sorted(preds.values(), key=lambda p: p["score"])
    assert rows[0]["rank"] <= rows[-1]["rank"]

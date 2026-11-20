"""Choosing which of a pose's own numbers the model may learn from.

The descriptor block ends with seven slots for the docking program's score
terms. Those slots were wired to DOCK 3's names and nothing else, so a mol2
from Glide, Vina, GOLD or rDock put zero in every one of them - and nothing
crashed, nothing warned, the model simply never saw the score its user
trusts most. That is the failure these tests pin.

The second thing pinned here is the junk filter. DOCK writes Number, Rank,
Setnum and Matchnum into the same comment block as the energies, all as
floats, and a failed term comes through as 999.999. A chooser that offers
those is worse than no chooser.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hitpicking.al import Learner  # noqa: E402
from hitpicking.al import scoreterms as S  # noqa: E402
from hitpicking.al.features import (SCORE_SLICE, TOTAL_BITS,  # noqa: E402
                                    descriptors, score_block)

HERE = os.path.dirname(__file__)
SAMPLE = ("/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty/"
          "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2")


def dock_samples(n=40):
    if not os.path.exists(SAMPLE):
        pytest.skip("the DOCK sample mol2 is not on this machine")
    from hitpicking.io.mol2 import read_mol2, dock_scores
    return [dock_scores(m) for m in read_mol2(SAMPLE, limit=n)]


# --------------------------------------------------------------- the junk

def test_a_row_number_is_not_a_score():
    samples = [{"Number": float(i), "Total Energy": -40.0 - i * 0.3}
               for i in range(30)]
    rows = {r["name"]: r for r in S.inspect(samples)}
    assert rows["Number"]["verdict"] == "identifier"
    assert not rows["Number"]["usable"]
    assert rows["Total Energy"]["usable"]


def test_a_constant_separates_nothing():
    samples = [{"Gist": 0.0, "Total Energy": -40.0 - i} for i in range(20)]
    rows = {r["name"]: r for r in S.inspect(samples)}
    assert rows["Gist"]["verdict"] == "constant"
    assert not rows["Gist"]["usable"]


def test_a_sentinel_is_not_a_measurement():
    # DOCK's own 999.999 for a term it could not evaluate. Left in, it
    # swamps the column mean and standard deviation for the whole set.
    samples = [{"Arbitrary": 999.999, "Electrostatic": -20.0 - i}
               for i in range(20)]
    rows = {r["name"]: r for r in S.inspect(samples)}
    assert not rows["Arbitrary"]["usable"]


def test_the_dock_file_yields_exactly_docks_own_terms():
    info = S.suggest(dock_samples())
    assert info["source"] == "dock"
    assert tuple(info["terms"]) == S.DOCK_TERMS
    rejected = {r["name"] for r in info["available"] if not r["usable"]}
    for junk in ("Number", "Rank", "Setnum", "Matchnum", "Arbitrary"):
        assert junk in rejected, "%s was offered as a score term" % junk


# ------------------------------------------------------ other programs

def test_smina_output_is_recognised():
    samples = [{"minimizedAffinity": -8.0 - i * 0.1,
                "minimizedRMSD": 0.4 + i * 0.01,
                "CNNscore": 0.5 + i * 0.005} for i in range(25)]
    info = S.suggest(samples)
    assert info["source"] == "vina"
    assert info["terms"][0] == "minimizedAffinity"


def test_an_unknown_program_still_gets_its_numbers_in():
    # The point of the generic path: no table entry, and it must work.
    samples = [{"MyScore": -30.0 - i, "MyVdw": -12.0 - i * 0.5,
                "pose_id": float(i), "always_seven": 7.0}
               for i in range(25)]
    info = S.suggest(samples)
    assert info["source"] == "generic"
    assert set(info["terms"]) == {"MyScore", "MyVdw"}
    assert "no docking program was recognised" in info["why"]


def test_nothing_usable_is_said_plainly_rather_than_guessed_at():
    info = S.suggest([{"Rank": 1.0} for _ in range(10)])
    assert info["terms"] == ()
    assert "no pose carries a usable score term" in info["why"]


def test_more_terms_than_slots_are_capped():
    names = ["term%d" % i for i in range(20)]
    samples = [{n: -1.5 * i - 0.25 * j for j, n in enumerate(names)}
               for i in range(20)]
    info = S.suggest(samples)
    assert len(info["terms"]) == S.N_SCORE_SLOTS


def test_an_integer_valued_score_is_not_mistaken_for_a_row_number():
    # The index heuristic must not fire on something that announces itself
    # as a score just because its values happen to be whole.
    samples = [{"Fitness Score": float(50 + i), "Number": float(i)}
               for i in range(25)]
    rows = {r["name"]: r for r in S.inspect(samples)}
    assert rows["Fitness Score"]["usable"]
    assert rows["Number"]["verdict"] == "identifier"


def test_normalize_is_what_the_model_records():
    assert S.normalize(["a", "a", "b"]) == ("a", "b")
    assert len(S.normalize(["t%d" % i for i in range(20)])) == S.N_SCORE_SLOTS
    assert S.normalize(None) == ()


# ------------------------------------------------- the bug being fixed

def test_a_non_dock_pose_scores_nothing_under_docks_terms():
    glide = {"r_i_docking_score": -9.4, "r_i_glide_evdw": -31.2}
    assert not np.any(score_block(glide, S.DOCK_TERMS))
    # ... and everything, once the terms match the file
    picked = S.suggest([glide, {"r_i_docking_score": -7.1,
                                "r_i_glide_evdw": -25.0}])["terms"]
    assert np.any(score_block(glide, picked))


def test_a_missing_term_leaves_its_slot_at_zero_not_at_nonsense():
    b = score_block({"Total Energy": -50.0}, S.DOCK_TERMS)
    assert b[0] == pytest.approx(-1.0)
    assert not np.any(b[1:])


# ----------------------------------------- changing the pick afterwards

class _Pose:
    """Just enough of a PoseResult for the descriptor block."""

    def __init__(self, scores):
        self.scores = scores
        self.combined = None
        self.counts = {}
        self.interactions = []
        self.strain = None


def _learner_with(samples, terms):
    lr = Learner(score_terms=terms)
    rng = np.random.default_rng(0)
    for i, sc in enumerate(samples):
        v = rng.normal(size=TOTAL_BITS)
        v[SCORE_SLICE] = score_block(sc, terms)
        lr.add_features("p%d" % i, v, sc)
        lr.grade("p%d" % i, i % 3)
    return lr


def test_changing_the_pick_rewrites_only_the_score_columns():
    samples = [{"Total Energy": -40.0 - i, "Max Strain": 1.0 + i * 0.1,
                "Electrostatic": -20.0 - i} for i in range(12)]
    lr = _learner_with(samples, S.DOCK_TERMS)
    before = {k: v.copy() for k, v in lr.features.items()}

    n = lr.set_score_terms(["Electrostatic", "Max Strain"])
    assert n == len(samples)
    assert lr.score_terms == ("Electrostatic", "Max Strain")

    for k, v in lr.features.items():
        rest_before = np.delete(before[k], np.s_[SCORE_SLICE])
        rest_after = np.delete(v, np.s_[SCORE_SLICE])
        assert np.array_equal(rest_before, rest_after), (
            "%s changed outside the score slots" % k)
        assert v[SCORE_SLICE][0] == pytest.approx(
            before[k][SCORE_SLICE][2])      # Electrostatic moved slot 2 -> 0


def test_changing_the_pick_discards_the_model_but_keeps_the_grades():
    samples = [{"Total Energy": -40.0 - i} for i in range(12)]
    lr = _learner_with(samples, S.DOCK_TERMS)
    lr.train(epochs=3)
    assert lr.committee is not None
    lr.set_score_terms(["Max Strain"])
    assert lr.committee is None, (
        "a committee fitted to different columns would predict nonsense")
    assert lr.trained_on == 0
    assert len(lr.grades) == len(samples)
    ok, _ = lr.can_train()
    assert ok


def test_setting_the_same_pick_is_a_no_op():
    lr = _learner_with([{"Total Energy": -40.0 - i} for i in range(12)],
                       S.DOCK_TERMS)
    lr.train(epochs=3)
    assert lr.set_score_terms(list(S.DOCK_TERMS)) == 0
    assert lr.committee is not None


# ------------------------------------------------------------ the model

def test_the_pick_travels_with_the_saved_model():
    lr = _learner_with([{"Total Energy": -40.0 - i} for i in range(12)],
                       ("Total Energy", "Max Strain"))
    st = lr.state()
    assert st["score_terms"] == ["Total Energy", "Max Strain"]
    back = Learner.from_state(st)
    assert back.score_terms == ("Total Energy", "Max Strain")


def test_a_model_saved_before_the_choice_existed_still_loads():
    lr = _learner_with([{"Total Energy": -40.0 - i} for i in range(12)],
                       S.DOCK_TERMS)
    st = lr.state()
    del st["score_terms"]                  # a file written by an older build
    back = Learner.from_state(st)
    assert back.score_terms == S.DOCK_TERMS, (
        "an old model was trained on DOCK's seven terms; it must load as "
        "that rather than be refused")


def test_the_vector_width_did_not_move():
    # Changing it would make every saved model unloadable, which is a far
    # bigger cost than this feature is worth.
    assert TOTAL_BITS == 1024 + 1024 + 21
    assert S.N_SCORE_SLOTS == 7
    assert descriptors(_Pose({"Total Energy": -50.0})).shape == (21,)

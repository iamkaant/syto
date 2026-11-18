"""Interaction detection: synthetic geometry, real poses, and invariants."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hitpicking.molsys import (MolSystem, HYB_SP2, HYB_SP3,  # noqa: E402
                               PART_LIGAND, PART_RECEPTOR)
from hitpicking.nci import criteria as C  # noqa: E402
from hitpicking.nci.detect import (detect_interactions, distance_score,  # noqa: E402
                                   angle_score)

AUTOPARTY = "/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty"
REC = os.path.join(AUTOPARTY, "autoparty/inputs/rec.crg.pdb")
LIG = os.path.join(AUTOPARTY, "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2")

needs_data = pytest.mark.skipif(not os.path.exists(REC),
                                reason="AutoParty sample data not available")


# ------------------------------------------------------------ score shapes

def test_distance_score_peaks_at_ideal():
    c = C.HBOND
    assert distance_score(c.d_ideal, c) == pytest.approx(1.0)
    assert distance_score(c.d_min - 0.01, c) == 0.0
    assert distance_score(c.d_max + 0.01, c) == 0.0
    assert 0 < distance_score(3.2, c) < 1


def test_angle_score_handles_both_directions():
    # larger-is-better (D-H...A -> 180)
    assert angle_score(180, 120, 180) == pytest.approx(1.0)
    assert angle_score(120, 120, 180) == 0.0
    # smaller-is-better (interplanar -> 0)
    assert angle_score(0, 30, 0) == pytest.approx(1.0)
    assert angle_score(30, 30, 0) == 0.0


def test_angle_score_returns_none_for_unmeasurable():
    assert angle_score(float("nan"), 120, 180) is None


# ------------------------------------------------- synthetic ideal geometry

def _hbond_system(d_oo=2.90, bend=0.0):
    """Ligand hydroxyl donating to a receptor carbonyl, D...A along +x.

    bend rotates the O-H away from linear so the D-H...A angle degrades.
    """
    a = math.radians(bend)
    xyz = [
        [0.0, 0.0, 0.0],                                   # 0 ligand O (donor)
        [0.96 * math.cos(a), 0.96 * math.sin(a), 0.0],      # 1 its H
        [-1.43, 0.0, 0.0],                                  # 2 ligand C
        [d_oo, 0.0, 0.0],                                   # 3 receptor O (acc)
        [d_oo + 1.23, 0.0, 0.0],                            # 4 receptor C
        [d_oo + 1.95, 1.05, 0.0],                           # 5 receptor C
    ]
    el = ["O", "H", "C", "O", "C", "C"]
    bonds = [[0, 1], [0, 2], [3, 4], [4, 5]]
    part = [PART_LIGAND] * 3 + [PART_RECEPTOR] * 3
    hyb = [HYB_SP3, HYB_SP3, HYB_SP3, HYB_SP2, HYB_SP2, HYB_SP3]
    return MolSystem(xyz, el, bonds, part=part, hyb=hyb,
                     name=["O1", "H1", "C1", "O", "C", "CB"],
                     res_name=["LIG", "LIG", "LIG", "ALA", "ALA", "ALA"])


def test_ideal_hbond_is_detected_with_high_quality():
    ms = _hbond_system(d_oo=C.HBOND.d_ideal, bend=0.0)
    res = detect_interactions(ms, enabled={"hbond"})
    hb = res.by_kind("hbond")
    assert len(hb) == 1, "one hydrogen bond expected, got %d" % len(hb)
    assert hb[0].distance == pytest.approx(C.HBOND.d_ideal, abs=1e-6)
    assert hb[0].angles["dha"] == pytest.approx(180.0, abs=0.5)
    assert hb[0].quality > 0.9


def test_bent_hbond_scores_lower_than_linear():
    # A 25 deg O-H rotation puts D-H...A near 144 deg: bent but still a real
    # hydrogen bond. (45 deg already drops it to ~118 deg, below the cutoff.)
    good = detect_interactions(_hbond_system(bend=0), enabled={"hbond"}).by_kind("hbond")
    bent = detect_interactions(_hbond_system(bend=25), enabled={"hbond"}).by_kind("hbond")
    assert good and bent
    assert 130 < bent[0].angles["dha"] < 150
    assert bent[0].quality < good[0].quality
    assert any("off linear" in f for f in bent[0].flags)


def test_hbond_beyond_cutoff_is_not_reported():
    ms = _hbond_system(d_oo=C.HBOND.d_max + 0.3)
    assert detect_interactions(ms, enabled={"hbond"}).by_kind("hbond") == []


def test_severely_bent_hbond_is_rejected():
    ms = _hbond_system(bend=60)       # D-H...A falls below the 120 deg cutoff
    assert detect_interactions(ms, enabled={"hbond"}).by_kind("hbond") == []


def _stacked_rings(sep=3.8, offset=1.5, tilt=0.0):
    """Two benzene rings, the second displaced along z by `sep`."""
    def ring(cx, cy, cz, tilt_deg):
        pts = []
        r = 1.39
        t = math.radians(tilt_deg)
        for k in range(6):
            a = 2 * math.pi * k / 6
            x, y, z = r * math.cos(a), r * math.sin(a), 0.0
            y, z = y * math.cos(t) - z * math.sin(t), y * math.sin(t) + z * math.cos(t)
            pts.append([cx + x, cy + y, cz + z])
        return pts
    p1 = ring(0, 0, 0, 0.0)
    p2 = ring(offset, 0, sep, tilt)
    xyz = p1 + p2
    bonds = [[k, (k + 1) % 6] for k in range(6)] + \
            [[6 + k, 6 + (k + 1) % 6] for k in range(6)]
    return MolSystem(xyz, ["C"] * 12, bonds,
                     hyb=[HYB_SP2] * 12, aromatic=[True] * 12,
                     part=[PART_LIGAND] * 6 + [PART_RECEPTOR] * 6,
                     res_name=["LIG"] * 6 + ["PHE"] * 6)


def test_parallel_stacking_detected_and_offset_preferred():
    res = detect_interactions(_stacked_rings(3.8, 1.5), enabled={"pi_stack_parallel"})
    st = res.by_kind("pi_stack_parallel")
    assert len(st) == 1
    assert st[0].angles["interplanar"] == pytest.approx(0.0, abs=1.0)
    assert st[0].quality > 0.8

    # An exactly eclipsed sandwich is real but less favourable, and is flagged.
    ecl = detect_interactions(_stacked_rings(3.8, 0.0),
                              enabled={"pi_stack_parallel"}).by_kind("pi_stack_parallel")
    assert ecl and any("offset" in f for f in ecl[0].flags)


def test_tshaped_stacking_detected():
    res = detect_interactions(_stacked_rings(5.0, 0.0, tilt=90.0),
                              enabled={"pi_stack_tshaped"})
    assert len(res.by_kind("pi_stack_tshaped")) == 1


def test_a_ring_pair_is_never_both_parallel_and_tshaped():
    ms = _stacked_rings(3.8, 1.5)
    res = detect_interactions(ms, enabled={"pi_stack_parallel", "pi_stack_tshaped"})
    assert len(res.interactions) == 1


# ------------------------------------------------------------- invariants

def test_toggles_are_honoured():
    ms = _hbond_system()
    assert detect_interactions(ms, enabled=set()).interactions == []
    assert detect_interactions(ms, enabled={"salt_bridge"}).by_kind("hbond") == []


def test_no_interaction_without_a_ligand():
    ms = _hbond_system()
    ms.part[:] = PART_RECEPTOR
    res = detect_interactions(ms)
    assert res.interactions == []
    assert res.warnings


def test_energy_estimate_scales_with_quality_and_is_never_summed():
    ms = _hbond_system()
    hb = detect_interactions(ms, enabled={"hbond"}).by_kind("hbond")[0]
    lo, typ, hi = hb.energy_estimate
    assert lo <= typ <= hi
    # it is a scaled range, not an absolute: quality 1 would give the full range
    assert typ <= C.HBOND.energy[1] + 1e-9


def test_every_criterion_cites_a_source():
    for key, crit in C.ALL.items():
        assert crit.refs, "criterion %s has no reference" % key
        assert crit.d_min < crit.d_ideal <= crit.d_max, key


# ------------------------------------------------------------- real poses

@needs_data
def test_known_opioid_pharmacology_is_recovered():
    """The MOR sample must reproduce its own design intent.

    The file is named 'mor_opcore_enum_asp_h2o_hbond': an opioid core
    enumeration built around an Asp interaction and structural waters. If the
    engine cannot find those, it is not working.
    """
    from hitpicking.analysis import ReceptorContext
    from hitpicking.io.mol2 import read_mol2
    ctx = ReceptorContext.from_pdb(REC)
    hits = {"asp": 0, "water": 0, "tyr_stack": 0}
    poses = read_mol2(LIG, limit=25)
    for p in poses:
        for it in ctx.analyze(p).interactions:
            if "ASP149" in it.label:
                hits["asp"] += 1
            if "HOH" in it.label:
                hits["water"] += 1
            if "TYR150" in it.label and it.kind.startswith("pi_stack"):
                hits["tyr_stack"] += 1
    assert hits["asp"] >= 20, hits
    assert hits["water"] >= 10, hits
    assert hits["tyr_stack"] >= 5, hits


@needs_data
def test_cached_receptor_matches_full_recompute():
    """The receptor-caching optimisation must not change any result."""
    from hitpicking.analysis import ReceptorContext
    from hitpicking.io.mol2 import read_mol2
    from hitpicking.perceive.features import perceive_all

    ctx = ReceptorContext.from_pdb(REC)

    def norm(interactions):
        return sorted(json.dumps({k: v for k, v in i.to_dict().items()
                                  if k != "label"}, sort_keys=True)
                      for i in interactions)

    for p in read_mol2(LIG, limit=12):
        fast = ctx.analyze(p)
        full = detect_interactions(ctx.combine(p),
                                   features=perceive_all(ctx.combine(p)))
        assert norm(fast.interactions) == norm(full.interactions), fast.name


@needs_data
def test_interactions_are_always_ligand_to_receptor():
    from hitpicking.analysis import ReceptorContext
    from hitpicking.io.mol2 import read_mol2
    ctx = ReceptorContext.from_pdb(REC)
    for p in read_mol2(LIG, limit=10):
        r = ctx.analyze(p)
        for it in r.interactions:
            assert it.lig_atoms and it.rec_atoms
            assert all(a >= r.n_receptor for a in it.lig_atoms), it.label
            assert all(a < r.n_receptor for a in it.rec_atoms), it.label


@needs_data
def test_results_are_json_serialisable():
    from hitpicking.analysis import ReceptorContext
    from hitpicking.io.mol2 import read_mol2
    ctx = ReceptorContext.from_pdb(REC)
    r = ctx.analyze(read_mol2(LIG, limit=1)[0])
    blob = json.dumps(r.to_dict())
    assert json.loads(blob)["name"]


# ------------------------------------- what distance does stacking measure?

def _stack(separation, offset, tilt=0.0):
    """Two rings with a chosen PLANE separation and lateral offset."""
    return _stacked_rings(sep=separation, offset=offset, tilt=tilt)


def test_parallel_stacking_scores_plane_separation_not_centroid_distance():
    """The literature 3.4-3.8 A for parallel stacking is between the PLANES.

    Scoring the centroid distance instead accepted rings 5 A apart as a good
    stack (a 5 A centroid separation sits inside a 5.5 A window) while
    rejecting a real 3.4 A stack that happened to be displaced by 3 A.
    """
    assert C.PI_STACK_PARALLEL.distance_measure == "interplanar_separation"

    # planes too far apart to stack, however they are aligned
    for sep in (5.0, 5.4, 6.0):
        res = detect_interactions(_stack(sep, 0.0),
                                  enabled={"pi_stack_parallel"})
        assert res.by_kind("pi_stack_parallel") == [], \
            "rings %.1f A apart must not be a parallel stack" % sep

    # a genuine stack that is laterally displaced is still a stack
    hits = detect_interactions(_stack(3.4, 3.0),
                              enabled={"pi_stack_parallel"}).by_kind("pi_stack_parallel")
    assert hits, "a displaced 3.4 A stack was missed"
    assert abs(hits[0].distance - 3.4) < 0.05, "reported distance is the separation"
    assert hits[0].angles["centroid"] > hits[0].distance, \
        "the centroid distance is reported alongside, and is larger"


def test_tshaped_stacking_uses_centroid_distance():
    """Perpendicular rings have no meaningful plane separation."""
    assert C.PI_STACK_TSHAPED.distance_measure == "centroid"
    hits = detect_interactions(_stack(0.0, 5.0, tilt=90.0),
                               enabled={"pi_stack_tshaped"}).by_kind("pi_stack_tshaped")
    assert hits
    assert abs(hits[0].distance - 5.0) < 0.1


def test_offset_ranking_matches_the_chemistry():
    """Parallel-displaced beats the sandwich, which is still a real contact."""
    def q(offset):
        hits = detect_interactions(_stack(3.4, offset),
                                   enabled={"pi_stack_parallel"}).by_kind("pi_stack_parallel")
        return hits[0].quality if hits else 0.0

    sandwich, ideal, wide = q(0.0), q(1.5), q(3.0)
    assert ideal > sandwich > 0, (sandwich, ideal)
    assert ideal > wide > 0, (wide, ideal)
    # the sandwich is penalised but not thrown away
    assert sandwich > 0.5


def test_sandwich_is_flagged_for_the_user():
    hits = detect_interactions(_stack(3.4, 0.0),
                               enabled={"pi_stack_parallel"}).by_kind("pi_stack_parallel")
    assert hits and any("sandwich" in f for f in hits[0].flags)


def test_every_criterion_declares_what_its_distance_means():
    for key, crit in C.ALL.items():
        assert crit.distance_measure in (
            "atoms", "centroid", "interplanar_separation"), (key,
                                                             crit.distance_measure)

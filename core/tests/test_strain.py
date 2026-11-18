"""Clash and strain detection: synthetic geometry with known answers."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hitpicking.molsys import (MolSystem, HYB_SP2, HYB_SP3,  # noqa: E402
                               PART_LIGAND, PART_RECEPTOR)
from hitpicking import strain as S  # noqa: E402

AUTOPARTY = "/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty"
REC = os.path.join(AUTOPARTY, "autoparty/inputs/rec.crg.pdb")
LIG = os.path.join(AUTOPARTY, "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2")
needs_data = pytest.mark.skipif(not os.path.exists(REC), reason="no sample data")


def two_atoms(d, el_a="C", el_b="C"):
    """One ligand atom and one receptor atom, `d` apart."""
    return MolSystem([[0, 0, 0], [d, 0, 0]], [el_a, el_b],
                     part=[PART_LIGAND, PART_RECEPTOR],
                     res_name=["LIG", "ALA"], name=["C1", "CB"])


# ------------------------------------------------------------- clash basics

def test_clash_threshold_matches_the_definition():
    """overlap = r1 + r2 - d, and must EXCEED the threshold."""
    rc = 1.70                      # Bondi carbon
    sep = 2 * rc - S.DEFAULT_CLASH_OVERLAP
    assert S.find_clashes(two_atoms(sep + 0.02)) == []     # just under
    hit = S.find_clashes(two_atoms(sep - 0.02))            # just over
    assert len(hit) == 1
    assert hit[0].overlap > S.DEFAULT_CLASH_OVERLAP


def test_hydrogen_bond_pairs_are_allowed_closer():
    """A donor/acceptor pair gets hbond_allowance before counting as a clash."""
    d = 2.55                       # N...O, close but a plausible H-bond
    assert S.find_clashes(two_atoms(d, "N", "O")) == []
    # the same distance between two carbons is a clash
    assert S.find_clashes(two_atoms(d, "C", "C"))


def test_severe_clashes_are_marked():
    rc = 1.70
    c = S.find_clashes(two_atoms(2 * rc - S.SEVERE_OVERLAP - 0.05))[0]
    assert c.severe
    mild = S.find_clashes(two_atoms(2 * rc - S.DEFAULT_CLASH_OVERLAP - 0.05))[0]
    assert not mild.severe


def test_no_ligand_means_no_clashes():
    ms = two_atoms(1.0)
    ms.part[:] = PART_RECEPTOR
    assert S.find_clashes(ms) == []


# --------------------------------------------------- bonded-pair exclusion

def _chain(n, spacing=1.5):
    xyz = [[i * spacing, 0, 0] for i in range(n)]
    bonds = [[i, i + 1] for i in range(n - 1)]
    return MolSystem(xyz, ["C"] * n, bonds,
                     part=[PART_LIGAND] * n, hyb=[HYB_SP3] * n)


def test_bonded_neighbours_never_clash():
    """Atoms 1.5 A apart overlap hugely but are bonded."""
    ms = _chain(5)
    assert S.find_internal_clashes(ms) == []


def test_bond_separation_matches_chimerax_default():
    """ChimeraX ignores pairs separated by <= 4 bonds; so must we.

    Using 3 flagged ordinary 1-5 contacts inside fused rings, which appeared
    identically in every pose of a series because they belong to the
    scaffold rather than the pose.
    """
    assert S.DEFAULT_BOND_SEPARATION == 4
    # a folded chain bringing atoms 0 and 4 (four bonds apart) together
    ms = _chain(5)
    ms.xyz[4] = [0.3, 0.0, 0.0]
    assert S.find_internal_clashes(ms) == []
    # five bonds apart, same contact, now counts
    ms2 = _chain(6)
    ms2.xyz[5] = [0.3, 0.0, 0.0]
    assert len(S.find_internal_clashes(ms2)) == 1


# ------------------------------------------------------- eclipsed torsions

def _butane(torsion_deg):
    """C1-C2-C3-C4 with a chosen C1-C2-C3-C4 dihedral."""
    t = math.radians(torsion_deg)
    xyz = [
        [-0.75, 1.23, 0.0],                                   # C1
        [0.0, 0.0, 0.0],                                      # C2
        [1.53, 0.0, 0.0],                                     # C3
        [2.28, 1.23 * math.cos(t), 1.23 * math.sin(t)],       # C4
    ]
    # two spectator heavy atoms so each inner carbon has >= 2 heavy neighbours
    xyz += [[-0.4, 2.2, 1.0], [3.3, 1.5, -0.7]]
    bonds = [[0, 1], [1, 2], [2, 3], [0, 4], [3, 5]]
    return MolSystem(xyz, ["C"] * 6, bonds,
                     part=[PART_LIGAND] * 6, hyb=[HYB_SP3] * 6,
                     res_name=["LIG"] * 6)


def _central(findings):
    """Only findings about the C2-C3 bond we actually set.

    The spectator atoms create their own sp3-sp3 bonds whose torsions are
    whatever the construction happens to give, and the detector is right to
    look at those too - so the test has to name the bond it means.
    """
    return [f for f in findings if f.atoms[1:3] == (1, 2)]


def test_eclipsed_torsion_is_detected():
    found = _central(S.find_eclipsed_torsions(_butane(2.0)))
    assert found, "a near-eclipsed sp3-sp3 torsion was not flagged"
    assert found[0].kind == "eclipsed_torsion"
    assert found[0].severity == "warning"


def test_staggered_torsion_is_not_flagged():
    assert _central(S.find_eclipsed_torsions(_butane(60.0))) == []
    assert _central(S.find_eclipsed_torsions(_butane(180.0))) == []


def test_gauche_eclipsed_at_120_is_detected():
    """120 deg is also an eclipsed rotamer, not just 0."""
    assert _central(S.find_eclipsed_torsions(_butane(118.0)))


def test_torsion_tolerance_is_respected():
    assert _central(S.find_eclipsed_torsions(_butane(35.0))) == []


# ------------------------------------------------------------ ring pucker

def _ring(pucker=0.0):
    n, r = 6, 1.39
    xyz = []
    for k in range(n):
        a = 2 * math.pi * k / n
        z = pucker if k % 2 == 0 else -pucker
        xyz.append([r * math.cos(a), r * math.sin(a), z])
    bonds = [[k, (k + 1) % n] for k in range(n)]
    return MolSystem(xyz, ["C"] * n, bonds, hyb=[HYB_SP2] * n,
                     aromatic=[True] * n, part=[PART_LIGAND] * n,
                     res_name=["LIG"] * n)


def test_flat_aromatic_ring_is_not_flagged():
    assert S.find_nonplanar_rings(_ring(0.0)) == []


def test_puckered_aromatic_ring_is_flagged():
    found = S.find_nonplanar_rings(_ring(0.45))
    assert found and found[0].kind == "nonplanar_ring"
    assert "flat" in found[0].message


# ---------------------------------------------------------- DOCK's own strain

def test_dock_strain_is_surfaced_not_recomputed():
    meta = {"dock": {"Total Strain": 11.0, "Max Strain": 4.2}}
    found = S.dock_strain_findings(meta)
    kinds = {f.kind for f in found}
    assert kinds == {"dock_total_strain", "dock_max_strain"}
    assert all(f.severity == "warning" for f in found)
    assert all(f.unit == "kcal/mol" for f in found)


def test_low_dock_strain_is_quiet():
    assert S.dock_strain_findings({"dock": {"Total Strain": 2.0,
                                            "Max Strain": 0.5}}) == []


def test_missing_dock_strain_is_not_invented():
    assert S.dock_strain_findings({}) == []
    assert S.dock_strain_findings(None) == []


# ------------------------------------------------------------- real poses

@needs_data
def test_strain_report_on_real_poses():
    from hitpicking.analysis import ReceptorContext
    from hitpicking.io.mol2 import read_mol2
    ctx = ReceptorContext.from_pdb(REC)
    seen_clash = False
    for p in read_mol2(LIG, limit=40):
        r = ctx.analyze(p)
        rep = S.analyze_strain(r.combined)
        assert rep.counts["clashes"] >= 0
        assert isinstance(rep.summary(), str)
        d = rep.to_dict()
        assert set(d) >= {"counts", "clashes", "internal_clashes", "findings"}
        if rep.clashes:
            seen_clash = True
            for c in rep.clashes:
                # a clash is always ligand-to-receptor
                assert r.combined.is_ligand[c.i]
                assert not r.combined.is_ligand[c.j]
    assert seen_clash, "no clash at all in 40 docked poses is implausible"

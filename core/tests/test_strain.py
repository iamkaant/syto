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


# ---------------------------------------------- conjugated torsions (aryl-N)

rdkit = pytest.importorskip("rdkit", reason="RDKit is a dev dependency")
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
RDLogger.DisableLog("rdApp.*")
from hitpicking.io.sdf import _parse_record  # noqa: E402
from hitpicking.strain import torsions as T  # noqa: E402


def build3d(smiles, seed=0xBEEF):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0, smiles
    AllChem.MMFFOptimizeMolecule(m)
    return _parse_record(Chem.MolToMolBlock(m).split("\n"), 1, smiles)


def set_torsion(ms, t, degrees):
    """Rotate everything on one side of the torsion's central bond."""
    p, a, b, q = t.atoms
    cur = ms.dihedral(p, a, b, q)
    delta = math.radians(degrees - cur)
    axis = ms.xyz[b] - ms.xyz[a]
    axis = axis / np.linalg.norm(axis)
    # atoms reachable from b without crossing the a-b bond
    moving, stack = set(), [b]
    while stack:
        cur_a = stack.pop()
        for nb in ms.neighbors(cur_a):
            nb = int(nb)
            if nb == a or nb in moving or nb == b:
                continue
            moving.add(nb)
            stack.append(nb)
    c, s_, t_ = math.cos(delta), math.sin(delta), 1 - math.cos(delta)
    x, y, z = axis
    R = np.array([
        [t_*x*x + c,   t_*x*y - s_*z, t_*x*z + s_*y],
        [t_*x*y + s_*z, t_*y*y + c,   t_*y*z - s_*x],
        [t_*x*z - s_*y, t_*y*z + s_*x, t_*z*z + c]])
    origin = ms.xyz[b].copy()
    for i in moving:
        ms.xyz[i] = origin + R @ (ms.xyz[i] - origin)


def _find(ms, subtype):
    for t in T.analyze_torsions(ms, ligand_only=False):
        if t.subtype == subtype:
            return t
    return None


def test_aniline_nitrogen_is_recognised():
    ms = build3d("Nc1ccccc1")          # aniline
    t = _find(ms, "aniline N")
    assert t is not None, "aryl-NH2 torsion not found"
    assert t.bond_type == T.CONJUGATED


def test_anilide_nitrogen_is_recognised():
    ms = build3d("CC(=O)Nc1ccccc1")    # acetanilide
    t = _find(ms, "anilide N")
    assert t is not None, "aryl-N(amide) torsion not found"


def test_coplanar_anilide_is_not_flagged():
    ms = build3d("CC(=O)Nc1ccccc1")
    t = _find(ms, "anilide N")
    set_torsion(ms, t, 0.0)
    again = _find(ms, "anilide N")
    assert not again.flagged, again.note


def test_perpendicular_anilide_is_flagged():
    """An anilide square to the ring has given up its conjugation."""
    ms = build3d("CC(=O)Nc1ccccc1")
    t = _find(ms, "anilide N")
    set_torsion(ms, t, 90.0)
    again = _find(ms, "anilide N")
    assert again.flagged, again.note
    # with the QM reference built this becomes a quantified finding, and
    # without it the geometric rule still catches it
    assert again.assessment in (T.TWISTED, T.STRAINED), again.assessment
    assert "ortho" in again.note or "kcal/mol" in again.note


def test_perpendicular_anilide_costs_several_kcal_when_qm_is_available():
    if T.qm_reference() is None:
        pytest.skip("QM torsion reference not built")
    ms = build3d("CC(=O)Nc1ccccc1")
    t = _find(ms, "anilide N")
    set_torsion(ms, t, 90.0)
    again = _find(ms, "anilide N")
    if again.energy is None:
        pytest.skip("no exact QM class for this bond")
    assert again.energy > 2.0, again.note
    # and coplanar must be cheap
    set_torsion(ms, t, 0.0)
    flat = _find(ms, "anilide N")
    assert flat.energy is None or flat.energy < again.energy


def test_ortho_substitution_is_detected_and_respected():
    """2,6-dimethylacetanilide is twisted because it must be, not from strain.

    The QM reference is keyed on ortho substitution for exactly this reason,
    and a class for the OTHER ortho state must never be used to flag: an
    ortho biaryl prefers ~38 degrees where an unsubstituted one prefers ~15,
    so judging one by the other would condemn the correct geometry.
    """
    ms = build3d("CC(=O)Nc1c(C)cccc1C")
    t = _find(ms, "anilide N")
    assert t is not None
    set_torsion(ms, t, 90.0)
    again = _find(ms, "anilide N")
    assert again.ortho_substituted, "ortho substituent not detected"
    if again.energy is not None:
        # an exact ortho class exists; it may legitimately say this is costly
        assert "indicative" not in again.note
    else:
        # no exact class: it must not be flagged from a mismatched curve
        assert not again.flagged or "indicative" not in again.note


def test_aryl_carbonyl_is_recognised():
    ms = build3d("O=C(C)c1ccccc1")     # acetophenone
    assert _find(ms, "aryl carbonyl") is not None


def test_biaryl_without_qm_is_not_guessed_at():
    """Both planar and twisted biphenyls are common.

    Without a reference the honest answer is to say so; with the QM curves
    the question is actually answerable, so the assertion is conditional.
    """
    ms = build3d("c1ccccc1-c1ccccc1")
    ts = [t for t in T.analyze_torsions(ms, ligand_only=False)
          if t.bond_type == T.BIARYL]
    assert ts, "biaryl bond not found"
    t = ts[0]
    if T.qm_reference() is None:
        assert t.assessment == T.UNJUDGED
        assert "CSD" in t.note
    else:
        assert "kcal/mol" in t.note, t.note


def test_qm_reference_knows_ortho_biaryls_prefer_a_twist():
    """The textbook case, and the one the geometric rules could not judge."""
    if T.qm_reference() is None:
        pytest.skip("QM torsion reference not built")
    flat_ortho = T.qm_cost("biaryl", "biaryl", True, 0.0)
    twist_ortho = T.qm_cost("biaryl", "biaryl", True, 40.0)
    if not (flat_ortho and twist_ortho and flat_ortho[5] and twist_ortho[5]):
        pytest.skip("no exact ortho biaryl class in this build")
    assert twist_ortho[0] < flat_ortho[0], (
        "an ortho-substituted biaryl should prefer a twist, got flat=%.2f "
        "twisted=%.2f" % (flat_ortho[0], twist_ortho[0]))


def test_unjudged_is_never_counted_as_fine():
    ms = build3d("c1ccccc1-c1ccccc1")
    ts = T.analyze_torsions(ms, ligand_only=False)
    for t in ts:
        if t.assessment == T.UNJUDGED:
            assert not t.flagged
            assert t.assessment != T.OK     # silence is not approval

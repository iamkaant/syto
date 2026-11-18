"""Core tests: data model, geometry, spatial search, parsers.

Run:  pytest core/tests -q
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hitpicking.molsys import MolSystem, HYB_SP2, HYB_SP3, PART_WATER  # noqa: E402
from hitpicking import elements as E  # noqa: E402

AUTOPARTY = "/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty"
LIG_SDF = os.path.join(AUTOPARTY, "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.sdf")
LIG_MOL2 = os.path.join(AUTOPARTY, "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2")
REC_PDB = os.path.join(AUTOPARTY, "autoparty/inputs/rec.crg.pdb")
REC_NOH = os.path.join(AUTOPARTY, "testing_inputs/D4.pdb")

needs_data = pytest.mark.skipif(
    not os.path.exists(LIG_SDF), reason="AutoParty sample data not available")


def benzene():
    n = 6
    r = 1.39 / (2 * math.sin(math.pi / n))
    xyz = [[r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), 0.0]
           for k in range(n)]
    bonds = [[k, (k + 1) % n] for k in range(n)]
    return MolSystem(xyz, ["C"] * n, bonds, hyb=[HYB_SP2] * n)


# --------------------------------------------------------------- data model

def test_ring_perception_benzene():
    ms = benzene()
    rings = ms.rings()
    assert len(rings) == 1
    assert len(rings[0]) == 6
    assert ms.ring_aromatic() == [True]


def test_ring_plane_normal_is_z():
    ms = benzene()
    _, nrm = ms.ring_plane(ms.rings()[0])
    assert abs(abs(nrm[2]) - 1.0) < 1e-9


def test_geometry_primitives():
    ms = MolSystem([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], ["C"] * 4,
                   [[0, 1], [1, 2], [2, 3]])
    assert abs(ms.distance(0, 1) - 1.0) < 1e-12
    assert abs(ms.angle(0, 1, 2) - 90.0) < 1e-9
    assert abs(abs(ms.dihedral(0, 1, 2, 3)) - 90.0) < 1e-9


def test_subset_remaps_bonds():
    ms = benzene()
    sub = ms.subset(np.array([True, True, True, False, False, False]))
    assert sub.n_atoms == 3
    assert len(sub.bonds) == 2
    assert sub.bonds.max() < 3


def test_adjacency_and_counts():
    ms = MolSystem([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   ["N", "H", "H", "C"], [[0, 1], [0, 2], [0, 3]])
    assert sorted(ms.neighbors(0).tolist()) == [1, 2, 3]
    assert ms.n_hydrogens()[0] == 2
    assert ms.heavy_degree()[0] == 1


@pytest.mark.parametrize("cutoff", [2.0, 4.0, 6.5])
def test_spatial_hash_matches_brute_force(cutoff):
    rng = np.random.default_rng(7)
    xyz = rng.uniform(-15, 15, (300, 3))
    ms = MolSystem(xyz, ["C"] * 300)
    i, j, _ = ms.pairs_within(cutoff)
    got = set(zip(i.tolist(), j.tolist()))
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    ii, jj = np.where((d <= cutoff) & np.triu(np.ones_like(d, dtype=bool), 1))
    assert got == set(zip(ii.tolist(), jj.tolist()))


def test_spatial_hash_masked_query_is_directional():
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-10, 10, (120, 3))
    ms = MolSystem(xyz, ["C"] * 120)
    a = np.zeros(120, bool); a[:20] = True
    b = np.zeros(120, bool); b[20:] = True
    i, j, _ = ms.pairs_within(5.0, a, b)
    assert i.max() < 20 and j.min() >= 20


def test_element_normalisation():
    assert E.normalize_element("C.ar") == "C"
    assert E.normalize_element("N.pl3") == "N"
    assert E.normalize_element(" Cl ") == "Cl"
    # 'CA' as an element field means calcium; as a PDB atom name it is a
    # carbon, which is why the PDB reader prefers columns 77-78.
    assert E.normalize_element("CA") == "Ca"


def test_vdw_estimates_are_flagged():
    for el in E.VDW_ESTIMATED:
        assert el in E.VDW_RADII, "estimated radius must still be present"
    assert "C" not in E.VDW_ESTIMATED


# ------------------------------------------------------------------- parsers

@needs_data
def test_mol2_parses_dock_score_block():
    from hitpicking.io.mol2 import read_mol2, pose_name, dock_scores
    poses = read_mol2(LIG_MOL2, limit=5)
    assert len(poses) == 5
    # the FIRST record must get its own comment block, not the second's
    assert pose_name(poses[0]) == "SD_295648_4650.0"
    s = dock_scores(poses[0])
    assert abs(s["Total Energy"] - (-53.131008)) < 1e-6
    assert "Van der Waals" in s and "Total Strain" in s
    # keys must be scalars, not lists -- a regression guard for comment leak
    assert isinstance(s["Total Energy"], float)


@needs_data
def test_mol2_names_are_distinct_per_record():
    from hitpicking.io.mol2 import read_mol2, pose_name
    names = [pose_name(p) for p in read_mol2(LIG_MOL2, limit=8)]
    assert len(set(names)) == len(names)


@needs_data
def test_sdf_reads_poses_and_props():
    from hitpicking.io.sdf import read_sdf
    mols = read_sdf(LIG_SDF, limit=10)
    assert len(mols) == 10
    assert all(m.n_atoms > 0 for m in mols)
    assert any(m.meta["props"] for m in mols)


@needs_data
def test_sulfonyl_sulfur_is_tetrahedral():
    """Regression: S=O bond orders must not make a sulfonyl S planar."""
    from hitpicking.io.sdf import read_sdf_iter
    found = False
    for m in read_sdf_iter(LIG_SDF):
        for i in range(m.n_atoms):
            if str(m.element[i]) != "S":
                continue
            o = sum(1 for j in m.heavy_neighbors(i) if str(m.element[j]) == "O")
            if o >= 2:
                found = True
                assert m.hyb[i] == HYB_SP3, "sulfonyl S typed as sp2"
        if found:
            break
    assert found, "no sulfonyl group in the sample"


@needs_data
def test_amide_nitrogen_is_conjugated():
    from hitpicking.io.sdf import read_sdf_iter
    from hitpicking.perceive.conjugation import is_amide_nitrogen
    hit = False
    for m in read_sdf_iter(LIG_SDF):
        conj = m.meta.get("conjugated")
        for i in range(m.n_atoms):
            if is_amide_nitrogen(m, i):
                hit = True
                assert conj[i], "amide N not flagged conjugated"
                assert m.hyb[i] == HYB_SP2, "amide N must be planar sp2"
        if hit:
            break
    assert hit, "no amide in the sample"


@needs_data
def test_pdb_reader_receptor():
    from hitpicking.io.pdb import read_pdb
    ms = read_pdb(REC_PDB)
    assert ms.n_atoms > 2000
    assert len(ms.bonds) > 2000
    assert (ms.part == PART_WATER).sum() > 0
    # histidine tautomers must survive as distinct residues
    assert {"HID", "HIE"} & set(ms.res_name.tolist())
    # aromatic side chains detected
    assert ms.aromatic.sum() > 100


@needs_data
def test_pdb_without_hydrogens_is_detectable():
    """D4.pdb has no H. Callers must be able to tell, because H-bond
    geometry and pKBHX relevance both depend on explicit hydrogens."""
    from hitpicking.io.pdb import read_pdb
    ms = read_pdb(REC_NOH)
    assert (ms.element == "H").sum() == 0


@needs_data
def test_no_metal_coordination_bonds():
    """Metals must not be covalently bonded; it would corrupt perception."""
    from hitpicking.io.pdb import read_pdb
    from hitpicking.elements import METALS
    ms = read_pdb(REC_PDB)
    if not len(ms.bonds):
        pytest.skip("no bonds")
    is_metal = np.array([str(e) in METALS for e in ms.element])
    assert not (is_metal[ms.bonds[:, 0]] | is_metal[ms.bonds[:, 1]]).any()


# ------------------------------------------------------- RDKit parity (dev)

rdkit = pytest.importorskip("rdkit", reason="RDKit is a dev-only dependency")


@needs_data
def test_perception_matches_rdkit():
    """The whole RDKit-free perception stack, checked against RDKit.

    Thresholds are set just above the measured rate (aromaticity 0.031%,
    hybridisation 0.006% over 54k atoms). Residual disagreements are exotic
    enumerated structures (radical carbons) where RDKit's valence model and
    our topology+geometry model legitimately differ.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from hitpicking.io.sdf import read_sdf_iter

    rd_hyb = {Chem.HybridizationType.SP: 1, Chem.HybridizationType.SP2: 2,
              Chem.HybridizationType.SP3: 3, Chem.HybridizationType.SP3D: 3,
              Chem.HybridizationType.SP3D2: 3}

    n_atom = arom_bad = hyb_bad = 0
    ours = read_sdf_iter(LIG_SDF)
    supp = Chem.ForwardSDMolSupplier(LIG_SDF, removeHs=False, sanitize=True)
    for k, (ms, rd) in enumerate(zip(ours, supp)):
        if rd is None or rd.GetNumAtoms() != ms.n_atoms:
            continue
        if k >= 400:
            break
        for i, at in enumerate(rd.GetAtoms()):
            n_atom += 1
            if at.GetIsAromatic() != bool(ms.aromatic[i]):
                arom_bad += 1
            h = rd_hyb.get(at.GetHybridization())
            if h is not None and at.GetSymbol() != "H" and h != int(ms.hyb[i]):
                hyb_bad += 1

    assert n_atom > 10000, "parity test needs a decent sample"
    assert arom_bad / n_atom < 0.002, "aromaticity drift: %d/%d" % (arom_bad, n_atom)
    assert hyb_bad / n_atom < 0.002, "hybridisation drift: %d/%d" % (hyb_bad, n_atom)

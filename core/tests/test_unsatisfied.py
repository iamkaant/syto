"""Buried polar atoms with no hydrogen-bond partner."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

from hitpicking.molsys import MolSystem, PART_LIGAND, PART_RECEPTOR  # noqa: E402
from hitpicking.nci import unsatisfied as U  # noqa: E402
from samples import RECEPTOR, POSES  # noqa: E402

needs_data = pytest.mark.skipif(not os.path.exists(RECEPTOR),
                                reason="no sample data")


def lone(el="O"):
    return MolSystem([[0.0, 0.0, 0.0]], [el], part=[PART_LIGAND],
                     res_name=["LIG"], name=["X1"])


def caged(el="O", shell=3.2):
    """One atom at the centre of a closed shell of carbons."""
    pts = U._sphere(60) * shell
    xyz = np.vstack([[0.0, 0.0, 0.0], pts])
    n = len(pts)
    return MolSystem(xyz, [el] + ["C"] * n,
                     part=[PART_LIGAND] + [PART_RECEPTOR] * n,
                     res_name=["LIG"] + ["ALA"] * n,
                     name=["X1"] + ["C%d" % i for i in range(n)])


def test_an_atom_alone_is_fully_exposed():
    r = 1.52 + U.PROBE
    assert U.accessible_area(lone(), [0]) == pytest.approx(
        4 * np.pi * r * r, rel=1e-6)


def test_an_atom_in_a_closed_cage_has_no_accessible_surface():
    assert U.accessible_area(caged(), [0]) == 0.0


def test_taking_the_cage_away_exposes_it_again():
    ms = caged()
    present = np.zeros(ms.n_atoms, bool)
    present[0] = True
    assert U.accessible_area(ms, [0], present=present) > 100.0


def test_no_polar_atoms_means_nothing_to_flag():
    assert U.find_unsatisfied(lone("C"), {"donors": [], "acceptors": []},
                              []) == []


@needs_data
def test_on_real_poses_the_flags_are_buried_unpaired_n_or_o():
    from hitpicking.analysis import ReceptorContext, Options
    from hitpicking.io.pdb import read_pdb
    from hitpicking.io.mol2 import read_mol2_iter
    ctx = ReceptorContext(read_pdb(RECEPTOR),
                          Options(stranded_acceptors=True))
    seen = flagged = 0
    for lig in read_mol2_iter(POSES):
        seen += 1
        if seen > 40:
            break
        res = ctx.analyze(lig)
        ms = res.combined
        paired = set()
        for it in res.interactions:
            if it.kind in U.PAIRING_KINDS:
                paired.update(it.lig_atoms)
                paired.update(it.rec_atoms)
        for u in res.unsatisfied:
            flagged += 1
            assert str(ms.element[u.atom]) in ("N", "O")
            assert u.atom not in paired, "flagged an atom that is H-bonded"
            assert u.sasa <= U.BURIED_SASA
            assert u.side == ("ligand" if ms.is_ligand[u.atom] else "receptor")
    assert flagged > 0, "nothing flagged on 40 poses: the check is not running"


@needs_data
def test_it_can_be_switched_off():
    from hitpicking.analysis import ReceptorContext, Options
    from hitpicking.io.pdb import read_pdb
    from hitpicking.io.mol2 import read_mol2_iter
    ctx = ReceptorContext(read_pdb(RECEPTOR),
                          Options(detect_unsatisfied=False))
    res = ctx.analyze(next(iter(read_mol2_iter(POSES))))
    assert res.unsatisfied is None
    assert "unsatisfied" not in res.to_dict()


@needs_data
def test_donors_are_reported_by_default_and_acceptors_are_not():
    from hitpicking.analysis import ReceptorContext, Options
    from hitpicking.io.pdb import read_pdb
    from hitpicking.io.mol2 import read_mol2_iter
    rec = read_pdb(RECEPTOR)
    default = ReceptorContext(rec, Options())
    both = ReceptorContext(rec, Options(stranded_acceptors=True))
    only_acc = ReceptorContext(rec, Options(stranded_donors=False,
                                            stranded_acceptors=True))
    n = roles_default = acc_seen = 0
    for lig in read_mol2_iter(POSES):
        n += 1
        if n > 40:
            break
        a = default.analyze(lig).unsatisfied
        b = both.analyze(lig).unsatisfied
        c = only_acc.analyze(lig).unsatisfied
        assert all("donor" in u.role for u in a)
        assert all("acceptor" in u.role for u in c)
        assert {u.atom for u in a} | {u.atom for u in c} == {u.atom for u in b}
        roles_default += len(a)
        acc_seen += len(c)
    assert roles_default > 0 and acc_seen > 0

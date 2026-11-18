"""Heteroaromatic ring identification, against known structures.

A specific wrong answer is worse than a vague right one: if the tool says
"pyrazine" it must be a pyrazine, because the basicity it then quotes differs
from pyridine by nearly a log unit. So these are built from SMILES with known
identities rather than inferred from a sample.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hitpicking.perceive.heteroaromatic import (  # noqa: E402
    classify_aromatic_nitrogen, order_ring, ring_separation,
    smallest_aromatic_ring)
from hitpicking.perceive.features import classify_acceptor, MOIETY_IDS  # noqa: E402

Chem = pytest.importorskip("rdkit.Chem", reason="RDKit is a dev dependency")
from rdkit import RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from hitpicking.io.sdf import _parse_record, RECORD_SEPARATOR  # noqa: E402


def build(smiles):
    """SMILES -> MolSystem, through a real SDF block (our normal input path)."""
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, smiles
    m = Chem.AddHs(m)
    assert AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE) == 0, smiles
    AllChem.MMFFOptimizeMolecule(m)
    block = Chem.MolToMolBlock(m)
    return _parse_record(block.split("\n"), 1, smiles)


# (SMILES, expected moiety id for the acceptor nitrogen(s))
CASES = [
    ("c1ccncc1",            "pyridine_n"),       # pyridine
    ("Cc1ccncc1",           "pyridine_n"),       # 4-methylpyridine
    ("c1ccc2ncccc2c1",      "quinoline_n"),      # quinoline
    ("c1ccc2cnccc2c1",      "isoquinoline_n"),   # isoquinoline
    ("c1cnccn1",            "pyrazine_n"),       # pyrazine   (1,4)
    ("c1cncnc1",            "pyrimidine_n"),     # pyrimidine (1,3)
    ("c1ccnnc1",            "pyridazine_n"),     # pyridazine (1,2)
    ("c1ncncn1",            "triazine_n"),       # 1,3,5-triazine
    ("Cn1ccnc1",            "imidazole_n3"),     # 1-methylimidazole
    ("Cn1cccn1",            "pyrazole_n2"),      # 1-methylpyrazole
    ("c1ccoc1" ,            None),               # furan: no N
    ("c1cnco1",             "oxazole_n3"),       # oxazole    (N,O 1,3)
    ("c1cnoc1",             "isoxazole_n2"),     # isoxazole  (N,O 1,2)
    ("c1cncs1",             "thiazole_n3"),      # thiazole
    ("Cn1cncn1",            "triazole_n"),       # 1-methyl-1,2,4-triazole
]


def acceptor_nitrogens(ms):
    out = []
    for i in range(ms.n_atoms):
        if str(ms.element[i]) != "N":
            continue
        res = classify_acceptor(ms, i)
        if res:
            out.append((i, res[0]))
    return out


@pytest.mark.parametrize("smiles,expected", CASES)
def test_ring_identified(smiles, expected):
    ms = build(smiles)
    found = acceptor_nitrogens(ms)
    if expected is None:
        assert not found, "%s should have no acceptor N, got %s" % (smiles, found)
        return
    assert found, "%s produced no acceptor nitrogen" % smiles
    ids = {mid for _, mid in found}
    assert ids == {expected}, "%s -> %s, expected %s" % (smiles, ids, expected)


def test_every_emitted_id_is_registered():
    """Perception must never emit an id the handbook build does not know."""
    for smiles, _ in CASES:
        ms = build(smiles)
        for _, mid in acceptor_nitrogens(ms):
            assert mid in MOIETY_IDS, "%s emitted unregistered id %r" % (smiles, mid)


def test_ring_ordering_is_a_real_cycle():
    ms = build("c1cnccn1")
    ring = smallest_aromatic_ring(ms, next(
        i for i in range(ms.n_atoms) if str(ms.element[i]) == "N"))
    order = order_ring(ms, ring)
    assert order is not None and len(order) == 6
    # consecutive atoms must be bonded, and the cycle must close
    for k in range(len(order)):
        a, b = order[k], order[(k + 1) % len(order)]
        assert b in [int(x) for x in ms.neighbors(a)]


def test_separation_is_symmetric_and_minimal():
    ms = build("c1cnccn1")
    ring = smallest_aromatic_ring(ms, 0)
    order = order_ring(ms, ring)
    a, b = order[0], order[3]
    assert ring_separation(order, a, b) == 3
    assert ring_separation(order, b, a) == 3
    assert ring_separation(order, order[0], order[5]) == 1


def test_pyrazine_is_not_called_pyridine():
    """The regression this was written for."""
    ms = build("c1cnccn1")
    ids = {mid for _, mid in acceptor_nitrogens(ms)}
    assert "pyridine_n" not in ids
    assert ids == {"pyrazine_n"}


# ---------------------------------------------------- parent-compound anchors

def test_each_ring_gets_its_own_family_value():
    """The regression this work started from.

    Picking a pyrazine N used to return pyridine, diazine, imidazole and
    azole data at once - a span of more than 1.5 log units presented as one
    answer. Each ring must now resolve to its own family, anchored on the
    parent compound where the data contains it.
    """
    from hitpicking.handbook import Handbook
    hb = Handbook.load_or_none()
    if hb is None:
        pytest.skip("handbook.json not built")

    expected = {
        "c1ccncc1": ("pyridine_n", 1.86),
        "c1cnccn1": ("pyrazine_n", 0.92),
        "c1cncnc1": ("pyrimidine_n", 1.07),
        "c1ccnnc1": ("pyridazine_n", 1.65),
        "c1ccc2ncccc2c1": ("quinoline_n", 1.89),
    }
    for smiles, (mid, value) in expected.items():
        ms = build(smiles)
        idx = next(i for i, m in acceptor_nitrogens(ms) if m == mid)
        rep = hb.representative_for([mid], "pKBHX")
        assert rep is not None, "%s has no parent-compound anchor" % mid
        assert abs(float(rep["value"]) - value) < 0.01, (mid, rep["value"])
        assert rep["ref"]["doi"] and rep["ref"]["locator"]
        label = hb.label_for_atom(ms, idx, "pKBHX")
        assert ("%.2f" % value) in label, (mid, label)

    # and the families really are far apart, which is why this matters
    pyz = hb.representative_for(["pyrazine_n"], "pKBHX")["value"]
    pyr = hb.representative_for(["pyridine_n"], "pKBHX")["value"]
    assert pyr - pyz > 0.8


def test_substituted_pyrrole_type_n_is_not_an_acceptor():
    """N-methylimidazole: N1 does not accept, N3 does."""
    from hitpicking.perceive.features import classify_non_acceptor
    ms = build("Cn1ccnc1")
    acceptors = acceptor_nitrogens(ms)
    assert len(acceptors) == 1, acceptors
    assert acceptors[0][1] == "imidazole_n3"
    others = [i for i in range(ms.n_atoms)
              if str(ms.element[i]) == "N" and i != acceptors[0][0]]
    assert others
    neg = classify_non_acceptor(ms, others[0])
    assert neg and "sextet" in neg[1]


# ------------------------------------------------- lumped-family narrowing

LUMPED = [
    # (SMILES, moiety, expected parent value, must NOT be confused with)
    ("Oc1ccccc1", "phenol_oh", -0.07, 1.02),   # phenol vs ethanol
    ("CCO",       "alcohol_oh", 1.02, -0.07),  # ethanol vs phenol
]


@pytest.mark.parametrize("smiles,mid,value,other", LUMPED)
def test_lumped_entries_are_narrowed_to_the_right_family(smiles, mid, value, other):
    """"Alcohol / water / phenol O" is one entry covering three families.

    A phenol is near zero on the pKBHX scale and an aliphatic alcohol near
    1.0; quoting the lumped range for either reads an order of magnitude
    wrong. The statistics shown must come from that moiety's own family.
    """
    from hitpicking.handbook import Handbook
    hb = Handbook.load_or_none()
    if hb is None:
        pytest.skip("handbook.json not built")

    ms = build(smiles)
    idx = next(i for i in range(ms.n_atoms)
               if str(ms.element[i]) == "O"
               and classify_acceptor(ms, i)
               and classify_acceptor(ms, i)[0] == mid)

    rep = hb.representative_for([mid], "pKBHX")
    assert rep and abs(float(rep["value"]) - value) < 0.01, (mid, rep)

    ann = [a for a in hb.for_atom(ms, idx).annotations
           if a["quantity"] == "pKBHX"]
    assert ann, mid
    a = ann[0]
    assert a.get("narrowed"), "%s was not narrowed to its own family" % mid
    # Narrowing must actually drop values: the lumped entry has 16, the
    # phenols 3. (The numeric ranges still overlap, and legitimately so -
    # trifluoroethanol is as weak an acceptor as a phenol - so overlap is
    # not the thing to assert on.)
    full = hb.entry(a["entry"])["summary_stats"]["pKBHX"]["n"]
    assert a["n"] < full, (mid, a["n"], full)
    # and the two families' parent compounds must be far apart
    assert abs(float(rep["value"]) - other) > 0.8, (mid, rep["value"], other)

    label = hb.label_for_atom(ms, idx, "pKBHX")
    assert ("%.2f" % value) in label, label


def test_phenol_and_alcohol_do_not_share_a_narrative_entry():
    from hitpicking.handbook import Handbook
    hb = Handbook.load_or_none()
    if hb is None:
        pytest.skip("handbook.json not built")
    ph = {e["id"] for e in hb.for_moiety("phenol_oh")}
    al = {e["id"] for e in hb.for_moiety("alcohol_oh")}
    assert "phenol_oh" in ph and "phenol_oh" not in al


def test_single_value_families_show_no_pointless_range():
    """"-0.40 (-0.4--0.4)" is noise, and the double minus is unreadable."""
    from hitpicking.handbook import Handbook
    hb = Handbook.load_or_none()
    if hb is None:
        pytest.skip("handbook.json not built")
    ms = build("c1ccoc1")
    idx = next(i for i in range(ms.n_atoms) if str(ms.element[i]) == "O")
    label = hb.label_for_atom(ms, idx, "pKBHX")
    assert label and "(" not in label, label
    assert "--" not in label

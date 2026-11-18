"""Handbook: data integrity, contextual lookup, and the citation guarantee."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hitpicking.handbook import Handbook  # noqa: E402
from hitpicking.perceive.features import MOIETY_IDS  # noqa: E402

AUTOPARTY = "/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty"
LIG_MOL2 = os.path.join(AUTOPARTY,
                        "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2")

hb = Handbook.load_or_none()
needs_hb = pytest.mark.skipif(
    hb is None, reason="handbook.json not built (run tools/build_data.py)")
needs_data = pytest.mark.skipif(not os.path.exists(LIG_MOL2),
                                reason="AutoParty sample data not available")


@needs_hb
def test_every_numeric_value_is_cited():
    """The core promise of this project: no number without a source.

    A value may be absent (pending curation), but a value that is PRESENT
    must name both a DOI and where in that document it was read.
    """
    missing = []
    for eid, e in hb.entries.items():
        for v in e.get("values") or []:
            if v.get("value") is None:
                continue
            ref = v.get("ref") or {}
            if not ref.get("doi") or not ref.get("locator"):
                missing.append("%s/%s" % (eid, v.get("quantity")))
    assert not missing, "uncited values: %s" % missing[:10]


@needs_hb
def test_handbook_has_real_pkbhx_data():
    e = hb.entry("hbond_pyridine_n")
    assert e is not None
    st = e.get("summary_stats", {})
    assert "pKBHX" in st, "pyridine entry carries no pKBHX values"
    assert st["pKBHX"]["n"] >= 10
    # pyridines span roughly 0-3 log units; a wider range means a parse error
    assert -2.0 < st["pKBHX"]["min"] < st["pKBHX"]["max"] < 6.0


@needs_hb
def test_acceptor_ordering_matches_chemistry():
    """Sanity check across families: the known basicity ordering must hold.

    Catches a mis-joined mapping, which would otherwise be invisible.
    """
    def med(eid):
        return hb.entry(eid)["summary_stats"]["pKBHX"]["median"]

    # amides accept better than esters; sulfoxides beat sulfones;
    # anilines are much weaker than aliphatic amines
    assert med("hbond_amide_carbonyl_o") > med("hbond_ester_lactone_carbonyl_o")
    assert med("hbond_sulfoxide_o") > med("hbond_sulfone_o")
    assert med("hbond_tertiary_amine_n") > med("hbond_aniline_n")
    assert med("hbond_n_oxide_o") > med("hbond_nitro_nitroso_o")


@needs_hb
def test_conflicts_are_surfaced_not_silently_resolved():
    """Disagreeing sources must stay visible."""
    cf = hb.conflicts()
    assert cf, "extraction reported conflicts; the build dropped them"
    assert hb.stats["n_conflicts"] == len(cf)
    for c in cf:
        assert c.get("ref", {}).get("doi")


@needs_hb
def test_mapping_targets_exist_and_keys_are_real_moieties():
    for mid, rec in hb.by_moiety.items():
        assert mid in MOIETY_IDS, "unmapped-from id %r" % mid
        for eid in rec.get("local", []) + rec.get("data", []):
            assert eid in hb.entries, "%s -> missing entry %s" % (mid, eid)


@needs_hb
def test_always_relevant_entries_resolve():
    assert hb.always_relevant()


@needs_hb
@needs_data
def test_click_an_atom_returns_entry_and_highlights_the_moiety():
    from hitpicking.io.mol2 import read_mol2
    from hitpicking.perceive.features import perceive_all
    p = read_mol2(LIG_MOL2, limit=1)[0]
    perceive_all(p)

    hits = [hb.for_atom(p, i) for i in range(p.n_atoms)]
    good = [h for h in hits if h]
    assert good, "no atom in a drug-like ligand produced a handbook hit"

    for h in good:
        assert h.title
        assert h.entries
        # highlighting must cover more than the clicked atom for group
        # moieties; it must always at least include the atom itself
        assert h.atom in h.highlight


@needs_hb
@needs_data
def test_ring_click_highlights_the_whole_ring():
    from hitpicking.io.mol2 import read_mol2
    from hitpicking.perceive.features import perceive_all, find_aromatic_rings
    p = read_mol2(LIG_MOL2, limit=1)[0]
    perceive_all(p)
    rings = find_aromatic_rings(p)
    assert rings
    ring = rings[0]
    hit = hb.for_atom(p, ring.atoms[0])
    for a in ring.atoms:
        assert a in hit.highlight


@needs_hb
@needs_data
def test_atom_labels_show_family_range_not_a_false_single_value():
    """A label must not imply more precision than perception supports."""
    from hitpicking.io.mol2 import read_mol2
    from hitpicking.perceive.features import perceive_all
    p = read_mol2(LIG_MOL2, limit=1)[0]
    perceive_all(p)
    labels = hb.annotate_system(p, "pKBHX")
    assert labels, "no pKBHX labels produced for a drug-like ligand"
    for text in labels.values():
        assert "pKBHX" in text
        # a range, or a single cited value, but never a bare unqualified mean
        assert ("(" in text and " to " in text) or "n=" in text \
            or text.count(".") == 1


@needs_hb
@needs_data
def test_interaction_click_returns_geometry_guidance():
    from hitpicking.analysis import ReceptorContext
    from hitpicking.io.mol2 import read_mol2
    rec = os.path.join(AUTOPARTY, "autoparty/inputs/rec.crg.pdb")
    if not os.path.exists(rec):
        pytest.skip("receptor not available")
    ctx = ReceptorContext.from_pdb(rec)
    res = ctx.analyze(read_mol2(LIG_MOL2, limit=1)[0])
    assert res.interactions
    hit = hb.for_interaction(res.interactions[0])
    assert hit.entries
    assert hit.highlight


@needs_hb
def test_flagged_values_are_counted():
    """Crystal-statistics and gas-phase numbers must not pass as measured."""
    s = hb.stats
    assert s["n_flagged_values"] > 0
    assert set(s["flags"]) <= {
        "crystal_statistics", "gas_phase_only", "value_conflict",
        "secondary_source", "secondary_citation", "read_from_image",
        "computed"}


# ------------------------------------------------------- the basicity scale

@needs_hb
def test_scale_places_the_clicked_moiety_in_context():
    """A lone number says nothing; the scale is the comparison."""
    sc = hb.scale("pKBHX", ["pyrazine_n"], same_element_only=True)
    assert sc, "no scale produced"
    assert sc["min"] < sc["max"]
    labels = [f["label"] for f in sc["families"]]
    assert len(labels) >= 5, labels
    hi = [f for f in sc["families"] if f["highlighted"]]
    assert len(hi) == 1 and "azine" in hi[0]["label"], labels
    # the marker is the parent compound's own value
    assert sc["marker"] and abs(sc["marker"]["value"] - 0.92) < 0.01
    # restricted to nitrogen families
    assert all(f["element"] in ("N", "") for f in sc["families"]), labels
    # every tick carries its compound, so the UI can label it on hover
    for f in sc["families"]:
        assert f["points"]
        assert all("compound" in p and "value" in p for p in f["points"])


@needs_hb
def test_scale_orders_families_sensibly():
    """Stronger acceptors first: the ordering is the point of the figure."""
    sc = hb.scale("pKBHX", [], same_element_only=False)
    labels = [f["label"] for f in sc["families"]]
    def pos(name):
        return next(i for i, l in enumerate(labels) if name in l)
    # amines and imidazole above pyridine; pyridine above the azines
    assert pos("imidazole N") < pos("pyridine N") < pos("azine N")
    # carbonyls: amide above ester
    assert pos("amide C=O") < pos("ester C=O")


@needs_hb
def test_scale_text_marks_the_selected_family():
    lines = hb.scale_text("pKBHX", ["imidazole_n3"], same_element_only=True)
    assert lines
    marked = [l for l in lines if l.startswith(">")]
    assert len(marked) == 1 and "imidazole" in marked[0]
    assert any("2.42" in l or "pKBHX =" in l for l in lines)


@needs_hb
def test_scale_handles_a_moiety_with_no_data():
    sc = hb.scale("pKBHX", ["hydrophobic"], same_element_only=True)
    assert sc is not None
    assert sc["marker"] is None
    assert not any(f["highlighted"] for f in sc["families"])

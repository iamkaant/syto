"""Handbook: data integrity, contextual lookup, and the citation guarantee."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hitpicking.handbook import Handbook  # noqa: E402
from hitpicking.perceive.features import MOIETY_IDS  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))
from samples import AUTOPARTY  # noqa: E402
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


# ---------------------------------------------------------------------------
# pKBH+ : the other question about the same atom
#
# pKBHX says how well an atom accepts a hydrogen bond. pKBH+ says whether it
# is holding a proton at all. Laurence et al. measured both and found they
# barely correlate across families, so the tool must present them as two
# scales and never let one stand in for the other.
# ---------------------------------------------------------------------------


@needs_hb
def test_pkbh_plus_scale_exists_and_covers_the_families_that_matter():
    sc = hb.scale("pKBH+", [], same_element_only=False)
    assert sc, "no pKBH+ scale produced"
    labels = [f["label"] for f in sc["families"]]
    for want in ("3° amine", "pyridine N", "aniline N",
                 "amide C=O", "nitrile N"):
        assert any(want == l for l in labels), (want, labels)
    # every family that has the data is laid out, not just the curated ones
    assert len(labels) >= 20, labels


@needs_hb
def test_pkbh_plus_is_ordered_by_what_the_axis_shows():
    """FAMILY_ORDER is a pKBHX reading order and is wrong here.

    A phosphine oxide is near the top of the pKBHX scale and near the bottom
    of the pKBH+ one. Sorting the pKa plot by the pKBHX order would put P=O
    above the amines and quietly mislead.
    """
    sc = hb.scale("pKBH+", [], same_element_only=False)
    meds = [f["median"] for f in sc["families"]]
    assert meds == sorted(meds, reverse=True), \
        [(f["label"], f["median"]) for f in sc["families"]]
    labels = [f["label"] for f in sc["families"]]
    assert labels.index("3° amine") < labels.index("P=O")
    # ... while pKBHX keeps its curated order, where P=O leads
    pk = [f["label"] for f in hb.scale("pKBHX", [],
                                       same_element_only=False)["families"]]
    assert pk.index("P=O") < pk.index("3° amine")


@needs_hb
def test_pkbh_plus_carries_the_physiological_reference_line():
    sc = hb.scale("pKBH+", [], same_element_only=False)
    lines = sc.get("reference_lines") or []
    assert len(lines) == 1 and abs(lines[0]["value"] - 7.4) < 1e-9, lines
    assert sc["min"] <= 7.4 <= sc["max"], (sc["min"], sc["max"])
    # pKBHX has no such line: there is no pH on that axis
    assert not (hb.scale("pKBHX", [], same_element_only=False)
                .get("reference_lines"))


@needs_hb
def test_the_two_scales_are_labelled_so_they_cannot_be_confused():
    a = hb.scale("pKBHX", [], same_element_only=False)
    b = hb.scale("pKBH+", [], same_element_only=False)
    assert a["unit"] and b["unit"] and a["unit"] != b["unit"]
    assert "4-fluorophenol" in a["unit"]
    assert "conjugate acid" in b["unit"]
    assert a["title"] != b["title"]
    # each plot says what it is not, by name rather than by implication:
    # "the hydrogen-bond scale" reads as if there were only one
    assert "pKBHX" in b["caption"] and "r2 = 0.22" in b["caption"]
    assert "pKBH+" in a["caption"]
    for cap in (a["caption"], b["caption"]):
        assert "the hydrogen-bond scale" not in cap, cap


@needs_hb
def test_protonation_verdict_reads_off_the_marked_value():
    from hitpicking.handbook.scale import protonation_verdict

    # pyridine, pKa 5.2: neutral at pH 7.4, so it accepts
    sc = hb.scale("pKBH+", ["pyridine_n"], same_element_only=False)
    assert sc["marker"] and abs(sc["marker"]["value"] - 5.2) < 0.01
    v = sc["verdict"]
    assert v["level"] == "neutral" and v["percent_protonated"] < 2.0
    assert "accepts" in v["text"]

    # an aliphatic amine at pKa 10.6 is a cation in the pocket
    v = protonation_verdict(10.6)
    assert v["level"] == "cation" and v["percent_protonated"] > 99.0
    assert "salt bridge" in v["text"]

    # right at the line, both forms are present and it is worth arguing about
    v = protonation_verdict(7.4)
    assert v["level"] == "mixed" and abs(v["percent_protonated"] - 50.0) < 0.1
    v = protonation_verdict(8.2)          # ~86%: still genuinely ambiguous
    assert v["level"] == "mixed"
    v = protonation_verdict(8.6)          # ~94%: call it a cation
    assert v["level"] == "cation"

    # pKBHX gets no verdict: a protonation state is not what it measures
    assert "verdict" not in hb.scale("pKBHX", ["pyridine_n"],
                                     same_element_only=False)


@needs_hb
def test_wide_axes_get_readable_ticks():
    """pKBH+ spans ~27 log units; one tick per unit is an unreadable smear."""
    from hitpicking.handbook.scale import tick_step
    assert tick_step(6.0, 8) == 1.0
    assert tick_step(27.0, 8) == 5.0
    assert tick_step(0.4, 8) <= 0.1
    lines = hb.scale_text("pKBH+", ["pyridine_n"],
                          same_element_only=False, width=46)
    assert lines
    header = lines[0]
    # labels on the axis must not run into each other
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", header)
    assert 3 <= len(nums) <= 9, header
    assert any("pH 7.4" in l for l in lines), lines[:4]
    assert any("protonated" in l for l in lines), lines[-4:]


@needs_hb
def test_pkbh_plus_only_speaks_when_it_overrides_pkbhx():
    """One scale is the default; the second has to earn each appearance.

    Measured over the real ligands of three saved sessions, a pKBH+ that
    says "neutral at pH 7.4" adds nothing to pKBHX and fired on 240 of 240
    checkable atoms in one of them. The case it exists for - the atom is a
    cation, so the pKBHX number is for a species that is not in the pocket
    - was 15% of atoms in another.
    """
    def ann(mids, q):
        for a in hb.annotations_for(mids):
            if a["quantity"] == q:
                return a
        return None

    # a neutral acceptor: pKBHX is the whole story
    a = ann(["pyridine_n"], "pKBH+")
    assert a is not None, "the value is still in the data layer"
    assert a["verdict"]["level"] == "neutral"
    assert a["changes_the_answer"] is False

    # a tertiary amine is a cation at pH 7.4: pKBHX 1.9 would call it a
    # good acceptor, and in the pocket it donates instead
    a = ann(["amine_n_tertiary"], "pKBH+")
    assert a is not None and a["changes_the_answer"] is True, a
    assert a["verdict"]["level"] == "cation"
    x = ann(["amine_n_tertiary"], "pKBHX")
    assert x["value"] if "value" in x else x["median"] > 1.0

    # pKBHX itself is never suppressed
    for mids in (["pyridine_n"], ["amine_n_tertiary"], ["amide_o"]):
        x = ann(mids, "pKBHX")
        assert x is not None and x["changes_the_answer"] is True, mids

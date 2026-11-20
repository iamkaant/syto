"""The basicity scale: where this atom sits among all the others.

A single number ("pKBHX 1.86") tells you almost nothing on its own. What a
picker actually wants is the comparison the Laurence SI draws: one axis, every
acceptor family laid along it, and your atom marked on it. Is this nitrogen a
good acceptor, or only good relative to other nitrogens? Would swapping it for
the ring next door gain or lose a log unit?

This builds that view's data. Rendering belongs to each front end: SVG in the
panel, text in the ChimeraX log.
"""

from __future__ import annotations

import math

# Families worth plotting, in reading order, with the short labels the axis
# uses. This list is also the FILTER: an entry that is not here is not a
# family and does not go on the plot.
#
# It used to be only an ordering, with anything unlisted appended under its
# full entry title. That put "pKBHX hydrogen-bond basicity scale: definition,
# range, classes" on the axis as though it were a moiety, with its class
# bounds (-0.96 to 5.46) stretching the axis to fit, and overflowed the label
# gutter with three lines of prose. Adding a new family now takes one line
# here, which is a fair price for an axis that only carries moieties.
FAMILY_ORDER = [
    ("hbond_carboxylate_o", "carboxylate O"),
    ("hbond_phosphoryl_o", "P=O"),
    ("hbond_n_oxide_o", "N-oxide"),
    ("hbond_sulfoxide_o", "S=O sulfoxide"),
    ("hbond_imidazole_n3", "imidazole N"),
    ("hbond_tertiary_amine_n", "3° amine"),
    ("hbond_secondary_amine_n", "2° amine"),
    ("hbond_primary_amine_n", "1° amine"),
    ("hbond_aniline_n", "aniline N"),
    ("hbond_amidine_guanidine_imine_n", "amidine/imine N"),
    ("hbond_pyridine_n", "pyridine N"),
    ("hbond_azine_n", "azine N"),
    ("hbond_azole_n", "azole N"),
    ("hbond_amide_carbonyl_o", "amide C=O"),
    ("hbond_vinylogous_amide_o", "vinylogous amide C=O"),
    ("hbond_lactam_pyridone_o", "lactam C=O"),
    ("hbond_urea_carbamate_imide_o", "urea/carbamate C=O"),
    ("hbond_ketone_carbonyl_o", "ketone C=O"),
    ("hbond_aldehyde_o", "aldehyde C=O"),
    ("hbond_ester_lactone_carbonyl_o", "ester C=O"),
    ("hbond_nitrile_n", "nitrile N"),
    ("hbond_push_pull_nitrile_n", "push-pull nitrile N"),
    ("hbond_ether_o", "ether O"),
    ("hbond_alcohol_phenol_o", "alcohol/phenol O"),
    ("hbond_sulfone_o", "sulfone O"),
    ("hbond_sulfonamide_o", "sulfonamide O"),
    ("hbond_thiocarbonyl_s", "C=S"),
    ("hbond_thioether_thiol_s", "thioether S"),
    ("hbond_nitro_nitroso_o", "nitro O"),
    ("hbond_haloalkane_x", "halogen"),
    ("hbond_pi_acceptor", "arene π"),
    ("hbond_furan_thiophene_pyrrole", "furan/thiophene"),
]

# ---------------------------------------------------------------------------
# What each scale actually measures.
#
# These two look alike and are not. pKBHX is a hydrogen-bond equilibrium
# constant measured against 4-fluorophenol in CCl4: how well this atom accepts
# a hydrogen bond. pKBH+ is the pKa of the conjugate acid in water: whether
# this atom is carrying a proton at all. Laurence et al. measured both and
# report that across families they barely correlate (r2 = 0.22, n = 232), so
# reading one off the other is a mistake the tool should not help anyone make.
#
# `order` says how to lay the families out. "curated" keeps FAMILY_ORDER,
# which was written as a descending pKBHX reading order. On any other scale
# that order is meaningless - a phosphine oxide is near the top for pKBHX and
# near the bottom for pKBH+ - so those sort by median instead.
# ---------------------------------------------------------------------------

QUANTITY_INFO = {
    "pKBHX": {
        "title": "Hydrogen-bond basicity",
        "symbol": "pKBHX",
        "unit": "log K (4-fluorophenol, CCl4, 298 K)",
        "means": "how strongly this atom accepts a hydrogen bond",
        "order": "curated",
        "caption": ("Acceptor strength only, and only for a NEUTRAL "
                    "acceptor. Whether the atom is carrying a proton - "
                    "which decides whether it accepts anything at all - is "
                    "the pKBH+ tab."),
    },
    "pKBH+": {
        "title": "Proton basicity",
        "symbol": "pKBH+",
        "unit": "pKa of the conjugate acid, water, 298 K",
        "means": "whether this atom is protonated at physiological pH",
        "order": "median",
        "reference_lines": [
            {"value": 7.4, "label": "pH 7.4",
             "note": "a group above this line is mostly protonated"},
        ],
        "caption": ("Not the same quantity as pKBHX. Both measure "
                    "basicity, but pKBHX is measured against a hydrogen-bond "
                    "donor (4-fluorophenol in CCl4) and pKBH+ against a "
                    "proton in water. Across families they correlate at only "
                    "r2 = 0.22 (n = 232), so a strong acceptor need not be a "
                    "strong base - which is why both are here."),
    },
}

PHYSIOLOGICAL_PH = 7.4


def quantity_info(quantity):
    return QUANTITY_INFO.get(quantity, {
        "title": quantity, "symbol": quantity, "unit": "",
        "means": "", "order": "median",
    })


def protonation_verdict(pka, ph=PHYSIOLOGICAL_PH):
    """What a pKBH+ means for the pose in front of you.

    The share protonated is the Henderson-Hasselbalch fraction. It is stated
    for the PARENT compound the handbook value came from: substituents move
    a pKa by several log units, so this orients the eye, it does not settle
    the ligand's state.
    """
    frac = 1.0 / (1.0 + 10.0 ** (ph - float(pka)))
    pct = 100.0 * frac
    # The bands are the pKa within about a log unit of the pH. Inside that
    # window neither form can be assumed, and that is exactly the case worth
    # arguing about at a party rather than deciding silently.
    if pct >= 99.0:
        verdict = ("protonated at pH %.1f - it donates and can make a salt "
                   "bridge; it is not an acceptor here" % ph)
        level = "cation"
    elif pct >= 90.0:
        verdict = ("mostly protonated at pH %.1f - treat it as a cation "
                   "unless the pocket says otherwise" % ph)
        level = "cation"
    elif pct > 10.0:
        verdict = ("both forms present at pH %.1f - the pocket decides, and "
                   "this is worth arguing about" % ph)
        level = "mixed"
    else:
        verdict = "neutral at pH %.1f - it accepts rather than donates" % ph
        level = "neutral"
    return {"ph": ph, "percent_protonated": round(pct, 1),
            "level": level, "text": verdict}


ELEMENT_OF = {
    "N": ("imidazole N", "3° amine", "2° amine", "1° amine",
          "amidine/imine N", "aniline N", "pyridine N", "azine N",
          "azole N", "nitrile N"),
}



def tick_step(span, max_ticks=8):
    """A round step (1, 2, 5, 10, ...) giving at most `max_ticks` labels."""
    span = abs(float(span)) or 1.0
    raw = span / float(max(1, max_ticks))
    mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for mult in (1, 2, 5, 10):
        if mult * mag >= raw:
            return mult * mag
    return 10.0 * mag


def _element_hint(label):
    for el, labels in ELEMENT_OF.items():
        if label in labels:
            return el
    if " O" in label or "C=O" in label or "P=O" in label or "S=O" in label:
        return "O"
    if " S" in label or "C=S" in label:
        return "S"
    return ""


def build_scale(handbook, quantity="pKBHX", moiety_ids=(), same_element_only=False,
                max_points_per_family=60):
    """Data for a one-axis comparison of every family carrying `quantity`.

    `moiety_ids` are the moieties the user clicked; their families are marked
    `highlighted`, and the parent compound's value becomes the marker.

    `same_element_only` restricts the view to families whose acceptor atom is
    the same element as the clicked one - "show me the other nitrogens".
    """
    moiety_ids = list(moiety_ids or ())

    # which entries does the clicked atom belong to?
    target_entries = set()
    for mid in moiety_ids:
        rec = handbook.by_moiety.get(mid) or {}
        target_entries.update(rec.get("data", []))

    marker = None
    for mid in moiety_ids:
        rep = (handbook.representative.get(mid) or {}).get(quantity)
        if rep and isinstance(rep.get("value"), (int, float)):
            marker = {"value": float(rep["value"]),
                      "compound": rep.get("compound"),
                      "entry": rep.get("entry"),
                      "flags": rep.get("flags") or []}
            break

    want_element = ""
    order = {eid: i for i, (eid, _) in enumerate(FAMILY_ORDER)}
    labels = dict(FAMILY_ORDER)

    families = []
    for eid, entry in handbook.entries.items():
        stats = (entry.get("summary_stats") or {}).get(quantity)
        if not stats:
            continue
        if eid not in labels:
            # Not a moiety family: a scale definition, or a set of worked
            # examples. Its numbers are real and stay in the entry list;
            # they are just not a row on this axis.
            continue
        label = labels[eid]
        points = []
        for v in entry.get("values") or []:
            if v.get("quantity") != quantity:
                continue
            val = v.get("value")
            if not isinstance(val, (int, float)):
                continue
            points.append({
                "value": float(val),
                "compound": v.get("compound") or "",
                "flagged": bool(set(v.get("flags") or []) & {
                    "value_conflict", "gas_phase_only", "secondary_source",
                    "secondary_citation", "read_from_image", "computed"}),
            })
        if not points:
            continue
        points.sort(key=lambda p: p["value"])
        if len(points) > max_points_per_family:
            step = len(points) / float(max_points_per_family)
            points = [points[int(i * step)] for i in range(max_points_per_family)]

        hi = eid in target_entries
        el = _element_hint(label)
        if hi and not want_element:
            want_element = el
        families.append({
            "entry": eid, "label": label, "element": el,
            "highlighted": hi,
            "n": stats["n"], "min": stats["min"], "max": stats["max"],
            "median": stats["median"],
            "points": points,
            "_order": order.get(eid, 500),
        })

    if same_element_only and want_element:
        families = [f for f in families
                    if f["element"] == want_element or f["highlighted"]]

    info = quantity_info(quantity)
    if info.get("order") == "curated":
        families.sort(key=lambda f: (f["_order"], -f["median"]))
    else:
        # FAMILY_ORDER is a descending pKBHX reading order and means nothing
        # on another axis, so lay those out by what the axis actually shows.
        families.sort(key=lambda f: -f["median"])
    for f in families:
        f.pop("_order", None)

    if not families:
        return None

    lo = min(f["min"] for f in families)
    hi = max(f["max"] for f in families)

    lines = []
    for rl in info.get("reference_lines") or ():
        lines.append(dict(rl))
        lo = min(lo, rl["value"])
        hi = max(hi, rl["value"])

    pad = max(0.2, (hi - lo) * 0.04)
    out = {
        "quantity": quantity,
        "title": info.get("title") or quantity,
        "symbol": info.get("symbol") or quantity,
        "means": info.get("means") or "",
        "unit": info.get("unit", ""),
        "caption": info.get("caption", ""),
        "reference_lines": lines,
        "min": round(lo - pad, 2),
        "max": round(hi + pad, 2),
        "families": families,
        "marker": marker,
        "element": want_element,
        "source": "10.1021/jm801331y",
    }
    if quantity == "pKBH+" and marker:
        out["verdict"] = protonation_verdict(marker["value"])
    return out


def text_scale(scale, width=46):
    """A compact text rendering, for the ChimeraX log."""
    if not scale:
        return []
    lo, hi = scale["min"], scale["max"]
    span = (hi - lo) or 1.0

    def col(v):
        return max(0, min(width - 1, int(round((v - lo) / span * (width - 1)))))

    out = []
    axis = [" "] * width
    ticks = []
    # One tick per unit is right for pKBHX (span ~6) and unreadable for
    # pKBH+ (span ~27): the numbers run together into a smear. Pick a step
    # that leaves at most ~8 labels whatever the span.
    step = tick_step(span, 8)
    t = float(step) * math.ceil(lo / float(step))
    while t <= hi + 1e-9:
        c = col(t)
        if 0 <= c < width:
            axis[c] = "|"
            ticks.append((c, ("%g" % round(t, 2))))
        t += step
    header = [" "] * width
    for c, s in ticks:
        for k, ch in enumerate(s):
            if c + k < width:
                header[c + k] = ch
    out.append("    %s   %s" % ("".join(header), scale["quantity"]))
    out.append("    %s" % "".join(axis))

    for rl in scale.get("reference_lines") or ():
        row = [" "] * width
        c = col(rl["value"])
        row[c] = "!"
        for k, ch in enumerate(" " + rl["label"]):
            if c + 1 + k < width:
                row[c + 1 + k] = ch
        out.append("    %s" % "".join(row))

    for f in scale["families"]:
        row = ["·"] * width
        a, b = col(f["min"]), col(f["max"])
        for c in range(a, b + 1):
            row[c] = "-"
        for p in f["points"]:
            row[col(p["value"])] = "|"
        for rl in scale.get("reference_lines") or ():
            c = col(rl["value"])
            if row[c] in ("\u00b7", "-"):
                row[c] = ":"
        mark = ">" if f["highlighted"] else " "
        line = "%s   %s  %-22s n=%d" % (mark, "".join(row), f["label"], f["n"])
        out.append(line)

    m = scale.get("marker")
    if m:
        pointer = [" "] * width
        pointer[col(m["value"])] = "^"
        out.append("    %s" % "".join(pointer))
        out.append("    %s = %.2f (%s)" % (scale["quantity"], m["value"],
                                           m.get("compound") or "this moiety"))
    v = scale.get("verdict")
    if v:
        out.append("    %.0f%% protonated: %s" % (v["percent_protonated"],
                                                  v["text"]))
        out.append("    (that is the parent compound's pKa - substituents "
                   "move it by log units)")
    if scale.get("caption"):
        out.append("    " + scale["caption"])
    return out

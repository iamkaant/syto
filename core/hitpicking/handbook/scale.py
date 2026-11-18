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

# Families worth showing on the scale, in a sensible reading order, with the
# short labels the axis uses. Anything not listed is still included under its
# entry title, after these.
FAMILY_ORDER = [
    ("hbond_phosphoryl_o", "P=O"),
    ("hbond_n_oxide_o", "N-oxide"),
    ("hbond_sulfoxide_o", "S=O sulfoxide"),
    ("hbond_imidazole_n3", "imidazole N"),
    ("hbond_tertiary_amine_n", "3° amine"),
    ("hbond_secondary_amine_n", "2° amine"),
    ("hbond_primary_amine_n", "1° amine"),
    ("hbond_amidine_guanidine_imine_n", "amidine/imine N"),
    ("hbond_pyridine_n", "pyridine N"),
    ("hbond_azine_n", "azine N"),
    ("hbond_azole_n", "azole N"),
    ("hbond_amide_carbonyl_o", "amide C=O"),
    ("hbond_lactam_pyridone_o", "lactam C=O"),
    ("hbond_urea_carbamate_imide_o", "urea/carbamate C=O"),
    ("hbond_ketone_carbonyl_o", "ketone C=O"),
    ("hbond_aldehyde_o", "aldehyde C=O"),
    ("hbond_ester_lactone_carbonyl_o", "ester C=O"),
    ("hbond_nitrile_n", "nitrile N"),
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

ELEMENT_OF = {
    "N": ("imidazole N", "3° amine", "2° amine", "1° amine",
          "amidine/imine N", "pyridine N", "azine N", "azole N", "nitrile N"),
}


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
        label = labels.get(eid) or entry.get("title", eid)
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

    families.sort(key=lambda f: (f["_order"], -f["median"]))
    for f in families:
        f.pop("_order", None)

    if not families:
        return None

    lo = min(f["min"] for f in families)
    hi = max(f["max"] for f in families)
    pad = max(0.2, (hi - lo) * 0.04)
    return {
        "quantity": quantity,
        "unit": "log K (4-fluorophenol, CCl4, 298 K)" if quantity == "pKBHX" else "",
        "min": round(lo - pad, 2),
        "max": round(hi + pad, 2),
        "families": families,
        "marker": marker,
        "element": want_element,
        "source": "10.1021/jm801331y",
    }


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
    t = int(lo)
    while t <= hi + 1e-9:
        c = col(t)
        if 0 <= c < width:
            axis[c] = "|"
            ticks.append((c, str(t)))
        t += 1
    header = [" "] * width
    for c, s in ticks:
        for k, ch in enumerate(s):
            if c + k < width:
                header[c + k] = ch
    out.append("    %s   %s" % ("".join(header), scale["quantity"]))
    out.append("    %s" % "".join(axis))

    for f in scale["families"]:
        row = ["·"] * width
        a, b = col(f["min"]), col(f["max"])
        for c in range(a, b + 1):
            row[c] = "-"
        for p in f["points"]:
            row[col(p["value"])] = "|"
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
    return out

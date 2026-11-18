"""Rotatable torsions: measure them all, judge only what can be judged.

The honest position
-------------------
Saying "this torsion costs 1.8 kcal/mol" requires CSD-derived torsion
distributions. The Hamburg/Roche Torsion Library states "All rights
reserved", its CSD-histogram version needs a CCDC licence, and RDKit's ETKDG
preferences are compiled into C++ rather than shipped as a reusable table.
There is therefore no torsion library here, and inventing one would be worse
than having none.

What is still worth doing, and is done here:

* measure EVERY rotatable torsion and show it, so a picker can look at the
  numbers the way they would in Mogul;
* flag the cases where the chemistry is unambiguous without any statistics:

    - an sp3-sp3 bond sitting on an eclipsed rotamer is on a barrier top;
    - a conjugated system (aryl-carbonyl, aryl-nitro, enone...) twisted far
      out of plane with NO ortho substituent to force it has given up its
      conjugation for nothing;
    - conversely, an ortho-substituted biaryl forced nearly flat is fighting
      a steric clash it should have relieved by twisting;
    - an amide twisted off planar.

* and mark everything else `unjudged`, explicitly, rather than implying the
  absence of a flag means the torsion is fine.
"""

from __future__ import annotations

import numpy as np

from ..molsys import HYB_SP, HYB_SP2, HYB_SP3

# A conjugated single bond twisted past this, with nothing forcing it, has
# lost conjugation it could have kept.
CONJUGATED_TWIST_LIMIT = 40.0
# An ortho-substituted biaryl flatter than this is sterically strained.
BIARYL_FORCED_PLANAR = 20.0
# Distance from an eclipsed rotamer (0 / 120 / 240) that counts as eclipsed.
ECLIPSED_TOLERANCE = 20.0

SP3_SP3 = "sp3-sp3"
CONJUGATED = "conjugated"
BIARYL = "biaryl"
AMIDE = "amide"
OTHER = "other"

# --------------------------------------------------------------------------
# QM reference, when it has been built (tools/fetch_qm_torsions.py then
# tools/build_torsion_profiles.py). Without it the thresholds below are the
# only judgement available; with it, a flag can quote what the twist costs.
# --------------------------------------------------------------------------

_QM = None
_QM_TRIED = False

# A torsion costing less than this is not worth mentioning; past the second
# value it is worth a warning rather than a note.
QM_NOTE_KCAL = 1.5
QM_WARN_KCAL = 3.0


def qm_reference():
    """The compiled QM torsion curves, or None if they were never built."""
    global _QM, _QM_TRIED
    if _QM_TRIED:
        return _QM
    _QM_TRIED = True
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "torsion_qm.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            _QM = json.load(fh)
    except Exception:
        _QM = None
    return _QM


def _qm_class(bond_type, subtype, ortho):
    qm = qm_reference()
    if not qm:
        return None
    key = "%s|%s|%s" % (bond_type, subtype or "-",
                        "ortho" if ortho else "no-ortho")
    cls = qm["classes"].get(key)
    if cls is not None:
        return cls, True
    # A class for the OTHER ortho state is not a substitute. An ortho
    # substituent is exactly what moves the preferred angle - an ortho biaryl
    # prefers about 38 degrees where an unsubstituted one prefers 15 - so
    # judging one by the other's curve would flag the correct geometry as
    # strained. Such a fallback is reported as indicative only, and never
    # raises a flag.
    for alt in ("%s|%s|%s" % (bond_type, subtype or "-",
                              "no-ortho" if ortho else "ortho"),
                "%s|-|no-ortho" % bond_type):
        cls = qm["classes"].get(alt)
        if cls is not None:
            return cls, False
    return None


def qm_cost(bond_type, subtype, ortho, angle):
    """What this torsion costs, in kcal/mol, or None if unknown.

    Returns (median, q1, q3, n_scans, class_name), interpolated between the
    binned angles of the matching class.
    """
    found = _qm_class(bond_type, subtype, ortho)
    if not found:
        return None
    cls, exact = found
    folded = _fold_for_class(angle, bond_type)
    curve = cls["curve"]
    xs = sorted(int(k) for k in curve)
    if not xs:
        return None
    if folded <= xs[0]:
        s = curve[str(xs[0])]
    elif folded >= xs[-1]:
        s = curve[str(xs[-1])]
    else:
        hi = next(x for x in xs if x >= folded)
        lo = max(x for x in xs if x <= folded)
        if hi == lo:
            s = curve[str(hi)]
        else:
            f = (folded - lo) / float(hi - lo)
            a, b = curve[str(lo)], curve[str(hi)]
            s = {k: a[k] + (b[k] - a[k]) * f for k in ("median", "q1", "q3")}
            s["n"] = min(a["n"], b["n"])
    name = "%s|%s|%s" % (cls["bond_type"], cls["subtype"] or "-",
                         "ortho" if cls["ortho_substituted"] else "no-ortho")
    return (s["median"], s["q1"], s["q3"], cls["n_scans"], name, exact)


def _fold_for_class(angle, bond_type):
    a = abs(float(angle)) % 360.0
    if a > 180.0:
        a = 360.0 - a
    if bond_type == SP3_SP3:
        f = a % 120.0
        return min(f, 120.0 - f)
    return min(a, 180.0 - a)


OK = "ok"
STRAINED = "strained"
ECLIPSED = "eclipsed"
TWISTED = "twisted out of conjugation"
FORCED_PLANAR = "forced planar"
UNJUDGED = "unjudged"


class Torsion:
    __slots__ = ("atoms", "angle", "bond_type", "assessment", "note",
                 "ortho_substituted", "subtype", "energy")

    def __init__(self, atoms, angle, bond_type, assessment, note="",
                 ortho_substituted=False, subtype=""):
        self.subtype = subtype
        # kcal/mol above this bond type's QM minimum, when the reference exists
        self.energy = None
        self.atoms = tuple(int(a) for a in atoms)
        self.angle = float(angle)
        self.bond_type = bond_type
        self.assessment = assessment
        self.note = note
        self.ortho_substituted = bool(ortho_substituted)

    @property
    def flagged(self):
        return self.assessment not in (OK, UNJUDGED)

    @property
    def bond(self):
        return (self.atoms[1], self.atoms[2])

    def __repr__(self):
        return "<Torsion %.0f %s %s>" % (self.angle, self.bond_type,
                                         self.assessment)

    def to_dict(self):
        return {"atoms": list(self.atoms), "angle": round(self.angle, 1),
                "bond_type": self.bond_type, "subtype": self.subtype,
                "energy_kcal": (None if self.energy is None
                                else round(self.energy, 2)),
                "assessment": self.assessment,
                "note": self.note, "flagged": self.flagged,
                "ortho_substituted": self.ortho_substituted}


def _ring_atoms(ms):
    out = set()
    for r in ms.rings():
        out.update(int(a) for a in r)
    return out


def _is_aromatic_carbon(ms, i):
    return bool(ms.aromatic[i])


def _conjugated_partner(ms, i):
    """True if atom i is sp2/sp and so can conjugate across a single bond."""
    return ms.hyb[i] in (HYB_SP, HYB_SP2)


def _has_ortho_substituent(ms, ring_atom, exclude):
    """Is either ring neighbour of `ring_atom` carrying a non-hydrogen?

    This is what decides whether a twisted conjugated system is strained or
    simply doing what sterics demand.
    """
    from ..perceive.heteroaromatic import smallest_aromatic_ring, order_ring
    ring = smallest_aromatic_ring(ms, ring_atom)
    if ring is None:
        return False
    order = order_ring(ms, ring)
    if order is None or ring_atom not in order:
        return False
    k = order.index(ring_atom)
    for nb in (order[(k + 1) % len(order)], order[(k - 1) % len(order)]):
        for sub in ms.heavy_neighbors(nb):
            sub = int(sub)
            if sub not in order and sub != exclude:
                return True
    return False


def _bears_double_bonded_o(ms, c):
    for o in ms.heavy_neighbors(c):
        o = int(o)
        if str(ms.element[o]) in ("O", "S", "N") \
                and len(ms.heavy_neighbors(o)) == 1 \
                and len(ms.attached_hydrogens(o)) == 0:
            return True
    return False


def conjugated_subtype(ms, a, b):
    """Name the conjugated system, because the chemistry differs by case.

    An aryl-amine and an aryl-ketone both prefer coplanarity, but for
    different reasons and with different costs, and a picker reads
    "aniline N twisted out of the ring plane" far better than
    "conjugated system twisted".
    """
    ar, other = (a, b) if _is_aromatic_carbon(ms, a) else (
        (b, a) if _is_aromatic_carbon(ms, b) else (None, None))
    if ar is None:
        return "conjugated"
    el = str(ms.element[other])
    if el == "N":
        from ..perceive.features import _is_amide_like
        if _is_amide_like(ms, other):
            return "anilide N"             # aryl-N of an amide
        n_o = sum(1 for x in ms.heavy_neighbors(other)
                  if str(ms.element[int(x)]) == "O")
        if n_o >= 2:
            return "aryl nitro"
        return "aniline N"                 # aryl-NR2
    if el == "C":
        if _bears_double_bonded_o(ms, other):
            has_n = any(str(ms.element[int(x)]) == "N"
                        for x in ms.heavy_neighbors(other))
            return "aryl amide C=O" if has_n else "aryl carbonyl"
        return "aryl conjugated C"
    if el == "O":
        return "aryl ether O"
    if el == "S":
        return "aryl S"
    return "conjugated"


def _classify_bond(ms, a, b):
    ar_a, ar_b = _is_aromatic_carbon(ms, a), _is_aromatic_carbon(ms, b)
    if ar_a and ar_b:
        return BIARYL
    if ms.hyb[a] == HYB_SP3 and ms.hyb[b] == HYB_SP3:
        return SP3_SP3
    # amide: N-C(=O)
    for x, y in ((a, b), (b, a)):
        if str(ms.element[x]) == "N" and str(ms.element[y]) == "C" \
                and _bears_double_bonded_o(ms, y):
            return AMIDE
    if _conjugated_partner(ms, a) and _conjugated_partner(ms, b):
        return CONJUGATED
    return OTHER


def rotatable_bonds(ms, ligand_only=True):
    """Single bonds worth measuring a torsion across."""
    rings = _ring_atoms(ms)
    out = []
    for a, b in ms.bonds:
        a, b = int(a), int(b)
        if ligand_only and not (ms.is_ligand[a] and ms.is_ligand[b]):
            continue
        if str(ms.element[a]) == "H" or str(ms.element[b]) == "H":
            continue
        if a in rings and b in rings:
            # a bond inside a ring is not freely rotatable; a bond BETWEEN
            # two rings is (biaryl), and those share no ring
            shared = False
            for r in ms.rings():
                members = set(int(x) for x in r)
                if a in members and b in members:
                    shared = True
                    break
            if shared:
                continue
        if not (_has_reference(ms, a, b) and _has_reference(ms, b, a)):
            continue
        out.append((a, b))
    return out


def _has_reference(ms, atom, across):
    """Can a torsion be measured from this end of the bond?

    Normally that needs a second heavy neighbour. An -NH2 or -OH has none,
    but its hydrogens still define the plane, and an aromatic amine twisted
    out of the ring plane is precisely the case worth catching - so a
    hydrogen counts as the reference on nitrogen and oxygen.

    Carbon is deliberately excluded: a methyl rotor would add a torsion to
    every molecule with nothing useful to say about it.
    """
    heavy = [int(x) for x in ms.heavy_neighbors(atom) if int(x) != across]
    if heavy:
        return True
    if str(ms.element[atom]) in ("N", "O"):
        return len(ms.attached_hydrogens(atom)) > 0
    return False


def _reference_neighbours(ms, a, b):
    """Pick the heaviest-substituted neighbour on each side, deterministically."""
    na = [int(x) for x in ms.heavy_neighbors(a) if int(x) != b]
    nb = [int(x) for x in ms.heavy_neighbors(b) if int(x) != a]
    # fall back to a hydrogen on N/O: see _has_reference
    if not na and str(ms.element[a]) in ("N", "O"):
        na = [int(x) for x in ms.attached_hydrogens(a)]
    if not nb and str(ms.element[b]) in ("N", "O"):
        nb = [int(x) for x in ms.attached_hydrogens(b)]
    if not na or not nb:
        return None, None
    na.sort(key=lambda i: (-len(ms.heavy_neighbors(i)), i))
    nb.sort(key=lambda i: (-len(ms.heavy_neighbors(i)), i))
    return na[0], nb[0]


def analyze_torsions(ms, ligand_only=True):
    """Every rotatable torsion, measured, with a judgement where possible."""
    out = []
    for a, b in rotatable_bonds(ms, ligand_only):
        p, q = _reference_neighbours(ms, a, b)
        if p is None:
            continue
        angle = ms.dihedral(p, a, b, q)
        btype = _classify_bond(ms, a, b)
        assessment, note = OK, ""
        ortho = False

        if btype == SP3_SP3:
            folded = abs(angle) % 120.0
            off = min(folded, 120.0 - folded)
            if off <= ECLIPSED_TOLERANCE:
                assessment = ECLIPSED
                note = ("%.0f deg is within %.0f deg of eclipsed; a staggered "
                        "rotamer is about 60 deg away" % (angle, off))

        elif btype in (CONJUGATED, BIARYL):
            twist = min(abs(angle), abs(180.0 - abs(angle)))
            ring_side = a if _is_aromatic_carbon(ms, a) else (
                b if _is_aromatic_carbon(ms, b) else None)
            if ring_side is not None:
                other = b if ring_side == a else a
                ortho = _has_ortho_substituent(ms, ring_side, other)
            sub = conjugated_subtype(ms, a, b) if btype == CONJUGATED else "biaryl"
            if btype == BIARYL:
                if ortho and twist < BIARYL_FORCED_PLANAR:
                    assessment = FORCED_PLANAR
                    note = ("an ortho-substituted biaryl held at %.0f deg is "
                            "fighting a steric clash it would normally relieve "
                            "by twisting" % twist)
                else:
                    assessment = UNJUDGED
                    note = ("biaryl twist %.0f deg. Both near-planar and "
                            "twisted conformers are common and the preference "
                            "needs CSD data, which is not available here - "
                            "judge by eye" % twist)
            elif twist > CONJUGATED_TWIST_LIMIT and not ortho:
                assessment = TWISTED
                note = ("%s is %.0f deg out of the ring plane with no ortho "
                        "substituent forcing it; coplanarity would keep the "
                        "conjugation" % (sub, twist))
            elif ortho:
                note = ("%s twisted %.0f deg, but there is an ortho "
                        "substituent - that is expected, not strain"
                        % (sub, twist))
            else:
                note = "%s, %.0f deg out of plane" % (sub, twist)

        elif btype == AMIDE:
            twist = min(abs(angle), abs(180.0 - abs(angle)))
            if twist > 25.0:
                assessment = TWISTED
                note = ("amide twisted %.0f deg from planar" % twist)

        else:
            assessment = UNJUDGED
            note = "no reliable preference available for this bond type"

        subtype = ""
        if btype in (CONJUGATED, BIARYL):
            subtype = conjugated_subtype(ms, a, b) if btype == CONJUGATED else "biaryl"
        elif btype == AMIDE:
            subtype = "amide C-N"
        # If the QM reference has been built, let it speak: it knows, for
        # instance, that an ortho-substituted biaryl PREFERS about 38 degrees,
        # which the geometric rules above could only call "unjudged".
        cost = qm_cost(btype, subtype, ortho, angle)
        if cost is not None:
            med, q1, q3, n, cls_name, exact = cost
            qm_note = ("%.1f kcal/mol above this bond type's minimum "
                       "(QM, %.1f-%.1f across %d scans)"
                       % (med, q1, q3, n))
            if not exact:
                qm_note += (" - indicative only: no QM class for this "
                            "ortho substitution, read from %s" % cls_name)
            if exact and med >= QM_NOTE_KCAL:
                assessment = STRAINED
                note = (note + "; " if note else "") + qm_note
            elif exact and assessment in (UNJUDGED, OK):
                assessment = OK
                note = qm_note
            else:
                note = (note + "; " if note else "") + qm_note
            energy = med if exact else None
        else:
            energy = None

        t = Torsion((p, a, b, q), angle, btype, assessment, note,
                    ortho, subtype)
        t.energy = energy
        out.append(t)
    return out


# ---------------------------------------------------------------- display


def profile_view(bond_type, subtype, ortho, angle=None):
    """The QM energy profile for this bond class, ready to plot.

    The analogue of the basicity scale: a single number says little, and the
    shape is the useful part. A flat-bottomed curve means the angle barely
    matters; a steep one means it does.
    """
    found = _qm_class(bond_type, subtype, ortho)
    if not found:
        return None
    cls, exact = found
    curve = []
    for k in sorted(cls["curve"], key=lambda x: int(x)):
        s = cls["curve"][k]
        curve.append({"angle": int(k), "median": s["median"],
                      "q1": s["q1"], "q3": s["q3"], "n": s["n"]})
    if not curve:
        return None
    marker = None
    if angle is not None:
        folded = _fold_for_class(angle, bond_type)
        cost = qm_cost(bond_type, subtype, ortho, angle)
        marker = {"angle": round(folded, 1),
                  "measured": round(float(angle), 1),
                  "energy": None if cost is None else round(cost[0], 2)}
    name = "%s|%s|%s" % (cls["bond_type"], cls["subtype"] or "-",
                         "ortho" if cls["ortho_substituted"] else "no-ortho")
    qm = qm_reference() or {}
    return {
        "class": name,
        "label": "%s%s" % (cls["subtype"] or cls["bond_type"],
                           ", ortho-substituted" if cls["ortho_substituted"]
                           else ""),
        "coordinate": cls.get("coordinate", ""),
        "x_label": ("degrees from eclipsed" if bond_type == SP3_SP3
                    else "degrees out of plane"),
        "n_scans": cls["n_scans"],
        "curve": curve,
        "marker": marker,
        "exact": exact,
        "barrier": cls.get("barrier", {}),
        "preferred_angle": cls.get("preferred_angle", {}),
        "units": "kcal/mol",
        "source": (qm.get("attribution", {}) or {}).get("source", ""),
    }


def profile_text(view, width=46, height=11):
    """An ASCII plot of a torsion profile, for the log.

    Same idea as the text basicity scale: the shape, with your angle marked.
    """
    if not view:
        return []
    curve = view["curve"]
    xs = [p["angle"] for p in curve]
    ys = [p["median"] for p in curve]
    lo_x, hi_x = min(xs), max(xs)
    hi_y = max(max(ys), 0.5)
    span_x = (hi_x - lo_x) or 1

    def cx(a):
        return int(round((a - lo_x) / span_x * (width - 1)))

    def cy(e):
        return int(round((1.0 - e / hi_y) * (height - 1)))

    grid = [[" "] * width for _ in range(height)]
    # interquartile band first, so the median line draws over it
    for p in curve:
        x = cx(p["angle"])
        y1, y2 = sorted((cy(p["q1"]), cy(p["q3"])))
        for y in range(y1, y2 + 1):
            if 0 <= y < height:
                grid[y][x] = ":"
    for p in curve:
        x, y = cx(p["angle"]), cy(p["median"])
        if 0 <= y < height:
            grid[y][x] = "*"

    marker = view.get("marker")
    mx = cx(marker["angle"]) if marker else None
    if mx is not None:
        for y in range(height):
            if grid[y][mx] == " ":
                grid[y][mx] = "|"

    out = []
    out.append("    %s  %s" % (view["label"],
                               "" if view["exact"] else "(indicative only)"))
    for y, row in enumerate(grid):
        energy = hi_y * (1.0 - y / float(height - 1))
        label = "%5.1f" % energy if y % 2 == 0 else "     "
        out.append("  %s |%s" % (label, "".join(row)))
    axis = ["-"] * width
    out.append("        +%s" % "".join(axis))
    ticks = [" "] * width
    for a in range(lo_x, hi_x + 1, 30):
        lab = str(a)
        # keep the last tick inside the plot: "90" was rendering as "9"
        x = min(cx(a), width - len(lab))
        for k, ch in enumerate(lab):
            if 0 <= x + k < width:
                ticks[x + k] = ch
    out.append("         %s" % "".join(ticks))
    out.append("         %s (kcal/mol on the left, %d scans)"
               % (view["x_label"], view["n_scans"]))
    if marker:
        e = marker["energy"]
        out.append("    measured %+.0f deg -> %s deg from the minimum%s"
                   % (marker["measured"], marker["angle"],
                      "" if e is None else ", costing %.1f kcal/mol" % e))
    return out


def torsion_summary(torsions):
    flagged = [t for t in torsions if t.flagged]
    unjudged = [t for t in torsions if t.assessment == UNJUDGED]
    return {"n": len(torsions), "flagged": len(flagged),
            "unjudged": len(unjudged),
            "by_type": {k: sum(1 for t in torsions if t.bond_type == k)
                        for k in (SP3_SP3, CONJUGATED, BIARYL, AMIDE, OTHER)
                        if any(t.bond_type == k for t in torsions)}}

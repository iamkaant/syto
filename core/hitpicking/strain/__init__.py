"""Clashes and internal strain in a docked pose.

What this does and does not claim
---------------------------------
Everything here is GEOMETRIC. It measures van der Waals overlap, torsion
angles, ring planarity and bond/angle deviation, and reports what is unusual.
It does not compute a strain energy.

That limit is deliberate, not an oversight. Strain energies worth quoting come
from CSD-derived torsion distributions (the Roche/Hamburg Torsion Library),
and that data is not redistributable: the dataset page states "All rights
reserved" and the version with CSD histograms needs a CCDC licence. Rather
than invent numbers, this reports the geometry and says how unusual it is in
plain terms. Where a user has their own licensed Mogul output, it can be
imported alongside.

A docked pose is also not a minimised structure. Mild deviations are normal
and the thresholds here are set to flag what a person would call wrong at a
glance, not every imperfection.
"""

from __future__ import annotations

import numpy as np

from ..elements import VDW_RADII, VDW_ESTIMATED, vdw_radius
from ..molsys import HYB_SP, HYB_SP2, HYB_SP3

__all__ = ["Clash", "StrainFinding", "atom_radii", "find_clashes", "find_internal_clashes",
           "find_eclipsed_torsions", "find_nonplanar_rings",
           "find_distorted_amides", "analyze_strain"]

# These match chimerax.clashes.settings.defaults exactly, so the parity test
# is meaningful and so a user comparing the two tools sees the same answer.
DEFAULT_CLASH_OVERLAP = 0.6        # clash_threshold
DEFAULT_HBOND_ALLOWANCE = 0.4      # clash_hbond_allowance
# Atoms separated by this many bonds or FEWER never clash. ChimeraX's default
# is 4; using 3 flagged ordinary 1-5 contacts in fused rings, which showed up
# identically in every pose of a series because they are a property of the
# scaffold, not of the pose.
DEFAULT_BOND_SEPARATION = 4
# A severe clash: something is badly wrong with the pose.
SEVERE_OVERLAP = 1.0

# sp3-sp3 torsions within this of perfectly eclipsed (0, 120, 240) are flagged.
ECLIPSED_TOLERANCE = 20.0
# An amide should be planar; past this it is strained or mis-built.
AMIDE_TWIST_TOLERANCE = 25.0
# Aromatic rings are flat; a docked pose puckered past this is suspect.
RING_PUCKER_TOLERANCE = 0.25


class Clash:
    __slots__ = ("i", "j", "distance", "overlap", "kind", "label", "severe")

    def __init__(self, i, j, distance, overlap, kind, label="", severe=False):
        self.i = int(i)
        self.j = int(j)
        self.distance = float(distance)
        self.overlap = float(overlap)
        self.kind = kind              # 'intermolecular' | 'intramolecular'
        self.label = label
        self.severe = bool(severe)

    def __repr__(self):
        return "<Clash %s %.2f overlap=%.2f>" % (self.kind, self.distance,
                                                 self.overlap)

    def to_dict(self):
        return {"atoms": [self.i, self.j], "distance": round(self.distance, 3),
                "overlap": round(self.overlap, 3), "kind": self.kind,
                "label": self.label, "severe": self.severe}


class StrainFinding:
    """One geometric oddity, described in words rather than scored."""

    __slots__ = ("kind", "atoms", "value", "unit", "message", "severity")

    def __init__(self, kind, atoms, value, unit, message, severity="note"):
        self.kind = kind
        self.atoms = tuple(int(a) for a in atoms)
        self.value = float(value)
        self.unit = unit
        self.message = message
        self.severity = severity      # 'note' | 'warning'

    def __repr__(self):
        return "<%s %.1f%s %s>" % (self.kind, self.value, self.unit,
                                   self.severity)

    def to_dict(self):
        return {"kind": self.kind, "atoms": list(self.atoms),
                "value": round(self.value, 2), "unit": self.unit,
                "message": self.message, "severity": self.severity}


# ---------------------------------------------------------------- clashes

def _bonded_within(ms, max_bonds=DEFAULT_BOND_SEPARATION):
    """Pairs separated by at most `max_bonds` bonds, which never clash.

    Same meaning as ChimeraX's `bond_separation`.
    """
    n = ms.n_atoms
    close = set()
    for i in range(n):
        frontier = {i}
        seen = {i}
        for _ in range(max_bonds):
            nxt = set()
            for a in frontier:
                for b in ms.neighbors(a):
                    b = int(b)
                    if b not in seen:
                        nxt.add(b)
                        seen.add(b)
            frontier = nxt
        for b in seen:
            if b != i:
                close.add((min(i, b), max(i, b)))
    return close


def _can_hbond(ms, i, j):
    """Crude donor/acceptor check, for the clash allowance only."""
    el_i, el_j = str(ms.element[i]), str(ms.element[j])
    polar = {"N", "O", "S", "F"}
    if el_i == "H":
        hv = ms.heavy_neighbors(i)
        el_i = str(ms.element[int(hv[0])]) if len(hv) else "C"
    if el_j == "H":
        hv = ms.heavy_neighbors(j)
        el_j = str(ms.element[int(hv[0])]) if len(hv) else "C"
    return el_i in polar and el_j in polar


def atom_radii(ms, radii=None):
    """Per-atom van der Waals radii for this system.

    A source that knows better supplies its own: the ChimeraX adapter passes
    through ChimeraX's per-IDATM-type radii, which are AMBER-style and
    noticeably larger than Bondi (iodine 2.80 vs 1.98, a methyl carbon 1.88
    vs 1.70). Using Bondi there made this tool report far fewer clashes than
    ChimeraX's own Clashes tool on the same pose, which is both wrong for hit
    picking and confusing to anyone comparing the two.

    Explicit `radii` wins, then whatever the source attached, then Bondi.
    """
    if radii is not None and not isinstance(radii, dict):
        return np.asarray(radii, dtype=np.float64)
    supplied = ms.meta.get("vdw_radii")
    if supplied is not None and radii is None:
        return np.asarray(supplied, dtype=np.float64)
    return np.array([vdw_radius(str(e), radii) for e in ms.element])


def find_clashes(ms, overlap_cutoff=DEFAULT_CLASH_OVERLAP,
                 hbond_allowance=DEFAULT_HBOND_ALLOWANCE,
                 radii=None, ligand_only=True, ignore_waters=False):
    """van der Waals overlaps between the ligand and everything else."""
    lig = ms.is_ligand
    if not lig.any():
        return []
    other = ~lig
    if ignore_waters:
        other = other & ~ms.is_water

    rad = atom_radii(ms, radii)
    probe = float(rad.max()) * 2 + 0.1
    ii, jj, dd = ms.pairs_within(probe, lig, other)
    out = []
    for i, j, d in zip(ii, jj, dd):
        i, j = int(i), int(j)
        ov = rad[i] + rad[j] - d
        if _can_hbond(ms, i, j):
            ov -= hbond_allowance
        if ov <= overlap_cutoff:
            continue
        out.append(Clash(i, j, d, ov, "intermolecular",
                         "%s / %s" % (ms.atom_label(i), ms.atom_label(j)),
                         severe=ov >= SEVERE_OVERLAP))
    out.sort(key=lambda c: -c.overlap)
    return out


def find_internal_clashes(ms, overlap_cutoff=DEFAULT_CLASH_OVERLAP,
                          hbond_allowance=DEFAULT_HBOND_ALLOWANCE,
                          radii=None, bond_separation=DEFAULT_BOND_SEPARATION):
    """The ligand clashing with itself: a sign of a strained conformer."""
    lig = ms.is_ligand
    if not lig.any():
        return []
    rad = atom_radii(ms, radii)
    probe = float(rad.max()) * 2 + 0.1
    ii, jj, dd = ms.pairs_within(probe, lig, lig)
    close = _bonded_within(ms, bond_separation)
    out = []
    seen = set()
    for i, j, d in zip(ii, jj, dd):
        i, j = int(i), int(j)
        key = (min(i, j), max(i, j))
        if key in seen or key in close:
            continue
        seen.add(key)
        ov = rad[i] + rad[j] - d
        if _can_hbond(ms, i, j):
            ov -= hbond_allowance
        if ov <= overlap_cutoff:
            continue
        out.append(Clash(i, j, d, ov, "intramolecular",
                         "%s / %s" % (ms.atom_label(i), ms.atom_label(j)),
                         severe=ov >= SEVERE_OVERLAP))
    out.sort(key=lambda c: -c.overlap)
    return out


# ----------------------------------------------------------------- strain

def _rotatable_sp3_bonds(ms):
    """Single bonds between two sp3 heavy atoms, not in a ring."""
    ring_atoms = set()
    for r in ms.rings():
        ring_atoms.update(int(a) for a in r)
    out = []
    for a, b in ms.bonds:
        a, b = int(a), int(b)
        if not (ms.is_ligand[a] and ms.is_ligand[b]):
            continue
        if str(ms.element[a]) == "H" or str(ms.element[b]) == "H":
            continue
        if ms.hyb[a] != HYB_SP3 or ms.hyb[b] != HYB_SP3:
            continue
        if a in ring_atoms and b in ring_atoms:
            continue
        if len(ms.heavy_neighbors(a)) < 2 or len(ms.heavy_neighbors(b)) < 2:
            continue
        out.append((a, b))
    return out


def find_eclipsed_torsions(ms, tolerance=ECLIPSED_TOLERANCE):
    """sp3-sp3 bonds sitting near an eclipsed rotamer.

    Staggered (60, 180, 300) is the minimum; eclipsed (0, 120, 240) is the
    barrier top. A docked pose resting on a barrier is usually an artefact of
    the search, and costs real energy the docking score may not have paid.
    """
    out = []
    for a, b in _rotatable_sp3_bonds(ms):
        na = [int(x) for x in ms.heavy_neighbors(a) if int(x) != b]
        nb = [int(x) for x in ms.heavy_neighbors(b) if int(x) != a]
        if not na or not nb:
            continue
        tor = ms.dihedral(na[0], a, b, nb[0])
        folded = abs(tor) % 120.0
        off = min(folded, 120.0 - folded)       # distance to an eclipsed value
        if off > tolerance:
            continue
        out.append(StrainFinding(
            "eclipsed_torsion", (na[0], a, b, nb[0]), tor, "deg",
            "%s-%s torsion is %.0f deg, within %.0f deg of eclipsed; "
            "staggered would be ~60 deg away"
            % (ms.atom_label(a), ms.atom_label(b), tor, off),
            severity="warning" if off <= tolerance / 2 else "note"))
    return out


def find_nonplanar_rings(ms, tolerance=RING_PUCKER_TOLERANCE):
    """Aromatic rings that are not flat.

    Aromaticity is decided from topology, so a puckered ring is reported here
    as a geometry problem rather than silently changing the chemistry.
    """
    out = []
    for ring, is_ar in zip(ms.rings(), ms.ring_aromatic()):
        if not is_ar or not all(ms.is_ligand[a] for a in ring):
            continue
        c, nrm = ms.ring_plane(ring)
        dev = float(np.abs((ms.xyz[list(ring)] - c) @ nrm).max())
        if dev <= tolerance:
            continue
        out.append(StrainFinding(
            "nonplanar_ring", ring, dev, "A",
            "aromatic ring is %.2f A out of plane; aromatic rings are flat, "
            "so this pose is distorted" % dev,
            severity="warning" if dev > tolerance * 2 else "note"))
    return out


def find_distorted_amides(ms, tolerance=AMIDE_TWIST_TOLERANCE):
    """Amide bonds twisted away from planar."""
    from ..perceive.features import _is_amide_like, _terminal_oxygens
    out = []
    for n in range(ms.n_atoms):
        if str(ms.element[n]) != "N" or not ms.is_ligand[n]:
            continue
        if not _is_amide_like(ms, n):
            continue
        for c in ms.heavy_neighbors(n):
            c = int(c)
            if str(ms.element[c]) != "C" or ms.hyb[c] != HYB_SP2:
                continue
            oxys = _terminal_oxygens(ms, c)
            if not oxys:
                continue
            subs = [int(x) for x in ms.heavy_neighbors(n) if int(x) != c]
            if not subs:
                continue
            tor = ms.dihedral(oxys[0], c, n, subs[0])
            twist = min(abs(tor), abs(180.0 - abs(tor)))
            if twist <= tolerance:
                continue
            out.append(StrainFinding(
                "twisted_amide", (oxys[0], c, n, subs[0]), twist, "deg",
                "amide at %s is twisted %.0f deg from planar; amides are "
                "planar and this costs real energy"
                % (ms.atom_label(n), twist),
                severity="warning"))
    return out


# -------------------------------------------------------------- top level

class StrainReport:
    def __init__(self, clashes, internal, findings, torsions=None):
        self.clashes = list(clashes)
        self.internal = list(internal)
        self.findings = list(findings)
        # every rotatable torsion, measured; most carry no judgement
        self.torsions = list(torsions or [])

    @property
    def flagged_torsions(self):
        return [t for t in self.torsions if t.flagged]

    @property
    def worst_overlap(self):
        allc = self.clashes + self.internal
        return max((c.overlap for c in allc), default=0.0)

    @property
    def counts(self):
        return {
            "clashes": len(self.clashes),
            "internal_clashes": len(self.internal),
            "severe_clashes": sum(1 for c in self.clashes + self.internal
                                  if c.severe),
            "strain_flags": len(self.findings),
            "torsions": len(self.torsions),
            "flagged_torsions": len(self.flagged_torsions),
        }

    def summary(self):
        c = self.counts
        bits = []
        if c["clashes"]:
            bits.append("%d clash%s with the site"
                        % (c["clashes"], "" if c["clashes"] == 1 else "es"))
        if c["internal_clashes"]:
            bits.append("%d internal clash%s"
                        % (c["internal_clashes"],
                           "" if c["internal_clashes"] == 1 else "es"))
        if c["strain_flags"]:
            bits.append("%d strain flag%s"
                        % (c["strain_flags"], "" if c["strain_flags"] == 1 else "s"))
        if c["flagged_torsions"]:
            bits.append("%d torsion%s worth a look"
                        % (c["flagged_torsions"],
                           "" if c["flagged_torsions"] == 1 else "s"))
        return "; ".join(bits) if bits else "no clashes or strain flags"

    def to_dict(self):
        return {"counts": self.counts,
                "worst_overlap": round(self.worst_overlap, 3),
                "clashes": [c.to_dict() for c in self.clashes],
                "internal_clashes": [c.to_dict() for c in self.internal],
                "findings": [f.to_dict() for f in self.findings],
                "torsions": [t.to_dict() for t in self.torsions]}


# DOCK 3.x writes its own torsional strain terms into the pose comment block.
# Those come from its internal library and are calibrated, unlike anything we
# could compute here without CSD data, so they are surfaced rather than
# re-derived. Thresholds are the values DISI uses as rules of thumb.
DOCK_TOTAL_STRAIN_WARN = 8.0
DOCK_MAX_STRAIN_WARN = 3.0


def dock_strain_findings(ligand_meta):
    """Flag high strain reported by DOCK itself, when the pose carries it."""
    dock = (ligand_meta or {}).get("dock") or {}
    out = []
    total = dock.get("Total Strain")
    mx = dock.get("Max Strain")
    if isinstance(total, float) and total >= DOCK_TOTAL_STRAIN_WARN:
        out.append(StrainFinding(
            "dock_total_strain", (), total, "kcal/mol",
            "DOCK reports total torsional strain %.1f (>= %.0f is usually "
            "worth a second look)" % (total, DOCK_TOTAL_STRAIN_WARN),
            severity="warning"))
    if isinstance(mx, float) and mx >= DOCK_MAX_STRAIN_WARN:
        out.append(StrainFinding(
            "dock_max_strain", (), mx, "kcal/mol",
            "DOCK reports a single torsion strained by %.1f (>= %.0f means "
            "one bond is carrying it)" % (mx, DOCK_MAX_STRAIN_WARN),
            severity="warning"))
    return out


def analyze_strain(ms, **kw):
    """Everything geometric that looks wrong about this pose."""
    clashes = find_clashes(ms, **{k: v for k, v in kw.items()
                                  if k in ("overlap_cutoff", "hbond_allowance",
                                           "radii", "ignore_waters")})
    internal = find_internal_clashes(
        ms, **{k: v for k, v in kw.items()
               if k in ("overlap_cutoff", "hbond_allowance", "radii",
                        "bond_separation")})
    from .torsions import analyze_torsions
    torsions = analyze_torsions(ms)

    # The torsion pass supersedes the standalone eclipsed/amide checks: it
    # covers the same bonds and also the conjugated ones, with the ortho
    # reasoning that decides whether a twist is strain or just sterics.
    findings = (find_nonplanar_rings(ms)
                + dock_strain_findings(ms.meta.get("ligand_meta")))
    for t in torsions:
        if not t.flagged:
            continue
        findings.append(StrainFinding(
            t.assessment.replace(" ", "_"), t.atoms, t.angle, "deg",
            t.note, severity="warning"))
    return StrainReport(clashes, internal, findings, torsions)

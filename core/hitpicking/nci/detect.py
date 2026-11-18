"""Non-covalent interaction detection and geometric scoring.

Every detector returns Interaction objects carrying not just "there is an
H-bond here" but *how good* it is: the measured distance and angles, how far
each is from the literature ideal, and a 0-1 quality. That is the part a
hit-picking party actually argues about.

Design rules
------------
* This is the ONE implementation. ChimeraX's own hbonds/clashes are used in
  the test suite as a cross-check, never at runtime, so the ChimeraX panel
  and the web page can never disagree.
* Scores are heuristic rankings, not free energies, and are never summed.
* Everything is toggleable, per interaction type.
"""

from __future__ import annotations

import numpy as np

from . import criteria as C
from ..molsys import HYB_SP2, HYB_SP3
from ..perceive.features import perceive_all

__all__ = ["Interaction", "detect_interactions", "InteractionSet"]


# ------------------------------------------------------------------ scoring

def distance_score(d, c):
    """1.0 at the ideal distance, falling to 0 at the limits."""
    if d < c.d_min or d > c.d_max:
        return 0.0
    if d <= c.d_ideal:
        span = c.d_ideal - c.d_min
        return 1.0 if span <= 0 else (d - c.d_min) / span
    span = c.d_max - c.d_ideal
    return 1.0 if span <= 0 else (c.d_max - d) / span


def angle_score(a, limit, ideal):
    """1.0 at the ideal angle, 0 at the cutoff. Handles both directions."""
    if a != a:                       # NaN: angle could not be measured
        return None
    if ideal >= limit:               # larger is better (e.g. D-H...A -> 180)
        if a <= limit:
            return 0.0
        if a >= ideal:
            return 1.0
        return (a - limit) / (ideal - limit)
    else:                            # smaller is better (e.g. interplanar -> 0)
        if a >= limit:
            return 0.0
        if a <= ideal:
            return 1.0
        return (limit - a) / (limit - ideal)


def _window_score(v, lo, ideal, hi):
    if v < lo or v > hi:
        return 0.0
    if v <= ideal:
        return 1.0 if ideal == lo else (v - lo) / (ideal - lo) if v > lo else 1.0
    return (hi - v) / (hi - ideal) if hi > ideal else 1.0


class Interaction:
    """One detected interaction, with its geometry and how good it is."""

    __slots__ = ("kind", "criterion", "lig_atoms", "rec_atoms", "distance",
                 "angles", "quality", "components", "flags", "moiety_ids",
                 "label", "water_mediated")

    def __init__(self, kind, criterion, lig_atoms, rec_atoms, distance,
                 angles=None, quality=0.0, components=None, flags=None,
                 moiety_ids=(), label="", water_mediated=False):
        self.kind = kind
        self.criterion = criterion
        self.lig_atoms = tuple(int(a) for a in lig_atoms)
        self.rec_atoms = tuple(int(a) for a in rec_atoms)
        self.distance = float(distance)
        self.angles = dict(angles or {})
        self.quality = float(quality)
        self.components = dict(components or {})
        self.flags = list(flags or [])
        self.moiety_ids = tuple(moiety_ids)
        self.label = label
        self.water_mediated = water_mediated

    @property
    def energy_estimate(self):
        """(low, typical, high) kcal/mol scaled by geometric quality.

        Explicitly an estimate. See criteria.py on why this is not a free
        energy and must not be summed.
        """
        e = self.criterion.energy
        if not e:
            return None
        return tuple(round(v * self.quality, 2) for v in e)

    def __repr__(self):
        return "<%s d=%.2f q=%.2f %s>" % (self.kind, self.distance,
                                          self.quality, self.label)

    def to_dict(self):
        return {
            "kind": self.kind, "label": self.label,
            "lig_atoms": list(self.lig_atoms), "rec_atoms": list(self.rec_atoms),
            "distance": round(self.distance, 3),
            "angles": {k: (None if v is None else round(v, 1))
                       for k, v in self.angles.items()},
            "quality": round(self.quality, 3),
            "components": {k: round(v, 3) for k, v in self.components.items()
                           if v is not None},
            "energy_estimate": self.energy_estimate,
            "flags": list(self.flags),
            "moiety_ids": list(self.moiety_ids),
            "curation_status": self.criterion.curation_status,
            "water_mediated": self.water_mediated,
        }


class InteractionSet:
    """Detected interactions plus the context needed to explain them."""

    def __init__(self, interactions, warnings=None, features=None, counts=None):
        self.interactions = list(interactions)
        self.warnings = list(warnings or [])
        self.features = features or {}
        self.counts = counts or {}

    def __len__(self):
        return len(self.interactions)

    def by_kind(self, kind):
        return [i for i in self.interactions if i.kind == kind]

    def to_dict(self):
        return {"interactions": [i.to_dict() for i in self.interactions],
                "warnings": self.warnings, "counts": self.counts}


# ---------------------------------------------------------------- detectors

def _hbond_angles(ms, d_idx, a_idx, h_idx):
    """(D-H...A, H...A-X) angles, or (nan, nan) when H is unavailable."""
    dha = float("nan")
    haa = float("nan")
    if h_idx is not None:
        dha = ms.angle(d_idx, h_idx, a_idx)
        nbrs = ms.heavy_neighbors(a_idx)
        if len(nbrs):
            haa = max(ms.angle(h_idx, a_idx, int(x)) for x in nbrs)
    else:
        nbrs = ms.heavy_neighbors(a_idx)
        if len(nbrs):
            haa = max(ms.angle(d_idx, a_idx, int(x)) for x in nbrs)
    return dha, haa


def detect_hbonds(ms, feats, lig_mask, rec_mask, include_weak=False,
                  water_as_receptor=True):
    """Hydrogen bonds between ligand and receptor (waters included)."""
    out = []
    has_h = feats["has_hydrogens"]
    donors = {f.root: f for f in feats["donors"]}
    acceptors = {f.root: f for f in feats["acceptors"]}
    charged_atoms = {a for f in feats["charged"] for a in f.atoms}

    d_idx = np.array(sorted(donors), dtype=np.int64)
    a_idx = np.array(sorted(acceptors), dtype=np.int64)
    if len(d_idx) == 0 or len(a_idx) == 0:
        return out

    dmask = np.zeros(ms.n_atoms, bool); dmask[d_idx] = True
    amask = np.zeros(ms.n_atoms, bool); amask[a_idx] = True

    crit_max = max(C.HBOND.d_max, C.HBOND_CHARGED.d_max,
                   C.WEAK_HBOND.d_max if include_weak else 0)

    for donor_side_is_lig in (True, False):
        dsel = dmask & (lig_mask if donor_side_is_lig else rec_mask)
        asel = amask & (rec_mask if donor_side_is_lig else lig_mask)
        if not dsel.any() or not asel.any():
            continue
        ii, jj, dd = ms.pairs_within(crit_max, dsel, asel)
        for d_i, a_i, dist in zip(ii, jj, dd):
            d_i, a_i = int(d_i), int(a_i)
            if d_i == a_i:
                continue
            if _are_bonded(ms, d_i, a_i):
                continue
            is_charged = d_i in charged_atoms or a_i in charged_atoms
            crit = C.HBOND_CHARGED if is_charged else C.HBOND
            if dist > crit.d_max:
                continue

            hyds = ms.attached_hydrogens(d_i) if has_h else []
            best = None
            for h in (list(hyds) if len(hyds) else [None]):
                h = None if h is None else int(h)
                dha, haa = _hbond_angles(ms, d_i, a_i, h)
                comp = {"distance": distance_score(dist, crit)}
                if comp["distance"] <= 0:
                    continue
                flags = []
                if h is not None:
                    s = angle_score(dha, *crit.angles["dha"])
                    if s is not None:
                        comp["dha"] = s
                        if s <= 0:
                            continue
                        if dha < 150:
                            flags.append("D-H...A %.0f deg, well off linear" % dha)
                s2 = angle_score(haa, *crit.angles["haa"])
                if s2 is not None:
                    comp["haa"] = s2
                vals = [v for v in comp.values() if v is not None]
                q = float(np.prod(vals)) ** (1.0 / len(vals)) if vals else 0.0
                if h is None:
                    flags.append("no explicit H: distance-only score")
                    q *= 0.8
                cand = (q, h, dist, {"dha": dha, "haa": haa}, comp, flags)
                if best is None or cand[0] > best[0]:
                    best = cand
            if best is None or best[0] <= 0:
                continue
            q, h, dist, angles, comp, flags = best

            lig_a = (d_i,) if donor_side_is_lig else (a_i,)
            rec_a = (a_i,) if donor_side_is_lig else (d_i,)
            if h is not None:
                if donor_side_is_lig:
                    lig_a = (d_i, h)
                else:
                    rec_a = (d_i, h)
            mids = (donors[d_i].moiety_id, acceptors[a_i].moiety_id)
            out.append(Interaction(
                crit.key, crit, lig_a, rec_a, dist, angles, q, comp, flags,
                mids,
                label="%s -> %s" % (ms.atom_label(d_i), ms.atom_label(a_i)),
            ))
    return out


def _are_bonded(ms, i, j):
    return j in set(int(x) for x in ms.neighbors(i))


def detect_salt_bridges(ms, feats, lig_mask, rec_mask):
    out = []
    crit = C.SALT_BRIDGE
    cats = [f for f in feats["charged"] if f.kind == "cation"]
    ans = [f for f in feats["charged"] if f.kind == "anion"]
    for a_grp in cats + ans:
        for b_grp in (ans if a_grp.kind == "cation" else cats):
            a_lig = all(lig_mask[x] for x in a_grp.atoms)
            b_lig = all(lig_mask[x] for x in b_grp.atoms)
            if a_lig == b_lig:
                continue                      # both sides same molecule
            best = None
            for x in a_grp.atoms:
                if str(ms.element[x]) == "C":
                    continue
                for y in b_grp.atoms:
                    if str(ms.element[y]) == "C":
                        continue
                    d = ms.distance(x, y)
                    if best is None or d < best[0]:
                        best = (d, x, y)
            if best is None:
                continue
            d, x, y = best
            q = distance_score(d, crit)
            if q <= 0:
                continue
            lig_atoms = a_grp.atoms if a_lig else b_grp.atoms
            rec_atoms = b_grp.atoms if a_lig else a_grp.atoms
            flags = []
            for g in (a_grp, b_grp):
                if "ASSUMED" in g.note or "assumed" in g.note:
                    flags.append(g.note)
            out.append(Interaction(
                crit.key, crit, lig_atoms, rec_atoms, d, {}, q,
                {"distance": q}, flags,
                (a_grp.moiety_id, b_grp.moiety_id),
                label="%s ... %s" % (ms.atom_label(x), ms.atom_label(y)),
            ))
    # de-duplicate the symmetric pass
    seen, uniq = set(), []
    for it in out:
        key = (tuple(sorted(it.lig_atoms)), tuple(sorted(it.rec_atoms)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    return uniq


def detect_pi_stacking(ms, feats, lig_mask, rec_mask):
    out = []
    rings = feats["rings"]
    lig_rings = [r for r in rings if all(lig_mask[a] for a in r.atoms)]
    rec_rings = [r for r in rings if all(rec_mask[a] for a in r.atoms)]
    for lr in lig_rings:
        c1, n1 = ms.ring_plane(lr.atoms)
        for rr in rec_rings:
            c2, n2 = ms.ring_plane(rr.atoms)
            v = c2 - c1
            d = float(np.linalg.norm(v))
            if d > max(C.PI_STACK_PARALLEL.d_max, C.PI_STACK_TSHAPED.d_max):
                continue
            cosang = abs(float(np.dot(n1, n2)))
            inter = float(np.degrees(np.arccos(min(1.0, max(-1.0, cosang)))))
            # lateral offset: component of the centroid vector in the ring plane
            along = abs(float(np.dot(v, n1)))
            offset = float(np.sqrt(max(d * d - along * along, 0.0)))

            for crit, use_offset in ((C.PI_STACK_PARALLEL, True),
                                     (C.PI_STACK_TSHAPED, False)):
                ds = distance_score(d, crit)
                if ds <= 0:
                    continue
                asc = angle_score(inter, *crit.angles["interplanar"])
                if asc is None or asc <= 0:
                    continue
                comp = {"distance": ds, "interplanar": asc}
                flags = []
                if use_offset:
                    lo, ideal, hi = C.PI_STACK_OFFSET
                    os_ = _window_score(offset, lo, ideal, hi)
                    comp["offset"] = os_
                    if os_ <= 0:
                        continue
                    if offset < 0.5:
                        flags.append("near-eclipsed sandwich; rings prefer "
                                     "~1.5 A offset")
                vals = list(comp.values())
                q = float(np.prod(vals)) ** (1.0 / len(vals))
                out.append(Interaction(
                    crit.key, crit, lr.atoms, rr.atoms, d,
                    {"interplanar": inter, "offset": offset}, q, comp, flags,
                    (lr.moiety_id, rr.moiety_id),
                    label="ring %s ... %s" % (
                        ms.atom_label(lr.atoms[0]), ms.atom_label(rr.atoms[0])),
                ))
                break        # a ring pair is parallel OR T-shaped, not both
    return out


def detect_cation_pi(ms, feats, lig_mask, rec_mask):
    out = []
    crit = C.CATION_PI
    rings = feats["rings"]
    cations = [f for f in feats["charged"] if f.kind == "cation"]
    for ring in rings:
        r_lig = all(lig_mask[a] for a in ring.atoms)
        c, nrm = ms.ring_plane(ring.atoms)
        for cat in cations:
            c_lig = all(lig_mask[a] for a in cat.atoms)
            if c_lig == r_lig:
                continue
            for x in cat.atoms:
                if str(ms.element[x]) == "C":
                    continue
                v = ms.xyz[x] - c
                d = float(np.linalg.norm(v))
                ds = distance_score(d, crit)
                if ds <= 0:
                    continue
                cosang = abs(float(np.dot(v / (d or 1.0), nrm)))
                off = float(np.degrees(np.arccos(min(1.0, max(-1.0, cosang)))))
                asc = angle_score(off, *crit.angles["normal_offset"])
                if asc is None or asc <= 0:
                    continue
                q = float(np.sqrt(ds * asc))
                lig_atoms = ring.atoms if r_lig else cat.atoms
                rec_atoms = cat.atoms if r_lig else ring.atoms
                out.append(Interaction(
                    crit.key, crit, lig_atoms, rec_atoms, d,
                    {"normal_offset": off}, q,
                    {"distance": ds, "normal_offset": asc},
                    [cat.note] if "assumed" in cat.note.lower() else [],
                    (ring.moiety_id, cat.moiety_id),
                    label="%s ... ring %s" % (ms.atom_label(x),
                                              ms.atom_label(ring.atoms[0])),
                ))
                break
    return out


def detect_halogen_bonds(ms, feats, lig_mask, rec_mask):
    out = []
    crit = C.HALOGEN_BOND
    acceptors = {f.root: f for f in feats["acceptors"]}
    if not acceptors:
        return out
    amask = np.zeros(ms.n_atoms, bool)
    for a in acceptors:
        amask[a] = True
    for hal in feats["halogens"]:
        x = hal.root
        side_lig = bool(lig_mask[x])
        sel = amask & (rec_mask if side_lig else lig_mask)
        if not sel.any():
            continue
        xm = np.zeros(ms.n_atoms, bool); xm[x] = True
        ii, jj, dd = ms.pairs_within(crit.d_max, xm, sel)
        cnb = ms.heavy_neighbors(x)
        if len(cnb) == 0:
            continue
        c = int(cnb[0])
        for _, a_i, dist in zip(ii, jj, dd):
            a_i = int(a_i)
            ds = distance_score(dist, crit)
            if ds <= 0:
                continue
            cxa = ms.angle(c, x, a_i)
            s1 = angle_score(cxa, *crit.angles["cxa"])
            if s1 is None or s1 <= 0:
                continue
            anb = ms.heavy_neighbors(a_i)
            xay = max((ms.angle(x, a_i, int(y)) for y in anb), default=float("nan"))
            s2 = angle_score(xay, *crit.angles["xay"])
            comp = {"distance": ds, "cxa": s1}
            if s2 is not None:
                comp["xay"] = s2
            vals = list(comp.values())
            q = float(np.prod(vals)) ** (1.0 / len(vals))
            flags = []
            if cxa < 155:
                flags.append("C-X...A %.0f deg; sigma hole is only weakly "
                             "aligned" % cxa)
            lig_a = (x,) if side_lig else (a_i,)
            rec_a = (a_i,) if side_lig else (x,)
            out.append(Interaction(
                crit.key, crit, lig_a, rec_a, dist,
                {"cxa": cxa, "xay": xay}, q, comp, flags,
                (hal.moiety_id, acceptors[a_i].moiety_id),
                label="%s ... %s" % (ms.atom_label(x), ms.atom_label(a_i)),
            ))
    return out


def detect_hydrophobic(ms, feats, lig_mask, rec_mask):
    out = []
    crit = C.HYDROPHOBIC
    hp = feats["hydrophobic"]
    a = hp & lig_mask
    b = hp & rec_mask
    if not a.any() or not b.any():
        return out
    ii, jj, dd = ms.pairs_within(crit.d_max, a, b)
    for i, j, d in zip(ii, jj, dd):
        q = distance_score(d, crit)
        if q <= 0:
            continue
        out.append(Interaction(
            crit.key, crit, (int(i),), (int(j),), d, {}, q, {"distance": q},
            [], ("hydrophobic", "hydrophobic"),
            label="%s ... %s" % (ms.atom_label(int(i)), ms.atom_label(int(j))),
        ))
    return out


DETECTORS = {
    "hbond": detect_hbonds,
    "salt_bridge": detect_salt_bridges,
    "pi_stack": detect_pi_stacking,
    "cation_pi": detect_cation_pi,
    "halogen_bond": detect_halogen_bonds,
    "hydrophobic": detect_hydrophobic,
}


def detect_interactions(ms, enabled=None, include_weak_hbonds=False,
                        features=None):
    """Detect every enabled interaction between the ligand and everything else.

    `enabled` is a set of criterion keys (see criteria.DEFAULT_ENABLED).
    Returns an InteractionSet.
    """
    enabled = set(C.DEFAULT_ENABLED if enabled is None else enabled)
    feats = features if features is not None else perceive_all(ms)

    lig_mask = ms.is_ligand.copy()
    rec_mask = ~lig_mask
    if not lig_mask.any():
        return InteractionSet([], ["No atoms are marked as ligand."], feats)

    out = []
    if {"hbond", "hbond_charged"} & enabled:
        out += detect_hbonds(ms, feats, lig_mask, rec_mask,
                             include_weak=include_weak_hbonds)
    if "salt_bridge" in enabled:
        out += detect_salt_bridges(ms, feats, lig_mask, rec_mask)
    if {"pi_stack_parallel", "pi_stack_tshaped"} & enabled:
        out += detect_pi_stacking(ms, feats, lig_mask, rec_mask)
    if "cation_pi" in enabled:
        out += detect_cation_pi(ms, feats, lig_mask, rec_mask)
    if "halogen_bond" in enabled:
        out += detect_halogen_bonds(ms, feats, lig_mask, rec_mask)
    if "hydrophobic" in enabled:
        out += detect_hydrophobic(ms, feats, lig_mask, rec_mask)

    out = [i for i in out if i.kind in enabled]
    out.sort(key=lambda i: -i.quality)

    counts = {}
    for i in out:
        counts[i.kind] = counts.get(i.kind, 0) + 1
    return InteractionSet(out, list(feats.get("warnings", [])), feats, counts)

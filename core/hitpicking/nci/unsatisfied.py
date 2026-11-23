"""Buried hydrogen-bond donors and acceptors with no partner.

"Watch out for buried polarity or missed opportunities" is on the
hit-picking checklist for a reason: a polar group that binding strips of
its water and then gives nothing back costs the full price of
desolvation. Bissantz, Kuhn & Stahl (J. Med. Chem. 2010, 53, 5061,
DOI 10.1021/jm100112j) put an unsatisfied buried donor or acceptor among
the most reliable reasons a pose is wrong.

Two conditions, both needed, and both heuristics:

  * no partner - the atom is in no detected hydrogen bond or salt
    bridge, and a deliberately LENIENT geometric test finds no
    complementary atom anywhere around it (the other molecule, its own
    molecule, a water). Lenient on purpose: this module accuses, so a
    poor hydrogen bond must still count as a hydrogen bond.
  * buried - a 1.4 A water probe cannot reach it in the complex. An
    exposed polar atom is paired with solvent that a docked pose simply
    does not show, and flagging those would bury the real cases. A
    pocket atom must also have been reachable BEFORE the ligand arrived:
    it is flagged only when this pose is what took its water away.

Only nitrogen and oxygen are judged.

The verdict is per atom, not per hydrogen or per lone pair: an NH3+
with one of three hydrogens paired is not flagged. That misses some
real problems and is the conservative side to be wrong on.
"""

from __future__ import annotations

import numpy as np

from ..elements import vdw_radius

PROBE = 1.4               # water, Angstrom
BURIED_SASA = 1.0         # A^2 of probe-accessible surface; below = buried
POCKET_RADIUS = 4.0       # receptor atoms this close to the ligand are judged
CONTACT_MAX = 3.5         # donor...acceptor heavy-atom distance, lenient
CONTACT_NO_H = 3.3        # the same, when there is no hydrogen to test
DHA_MIN = 110.0           # D-H...A angle, lenient
METAL_MAX = 2.8           # acceptor...metal
SPHERE_POINTS = 96

METALS = frozenset(("Li", "Na", "K", "Mg", "Ca", "Mn", "Fe", "Co", "Ni",
                    "Cu", "Zn", "Cd", "Hg", "Mo"))
# Acceptors in name only: a lone pair that is delocalised into a ring, a
# carbonyl, a sulfonyl or a nitro group accepts so weakly (pKBHX around
# zero or below on the Laurence scale, DOI 10.1021/jm801331y) that
# leaving it unpaired costs next to nothing. They are still drawn when
# they do make a contact; they are just never accused.
WEAK_ACCEPTORS = frozenset((
    "nitro_o", "aniline_n", "furan_o", "pyrrole_type_n_substituted",
    "amide_n_tertiary", "sulfonamide_n"))

PAIRING_KINDS = frozenset(("hbond", "hbond_charged", "salt_bridge",
                           "halogen_bond"))


class Unsatisfied:
    """One buried polar atom with nothing to hydrogen-bond to."""

    __slots__ = ("atom", "role", "side", "sasa", "label")

    def __init__(self, atom, role, side, sasa, label):
        self.atom = int(atom)
        self.role = role          # 'donor' | 'acceptor' | 'donor/acceptor'
        self.side = side          # 'ligand' | 'receptor'
        self.sasa = float(sasa)
        self.label = label

    @property
    def message(self):
        return ("%s: buried %s with no hydrogen-bond partner"
                % (self.label, self.role))

    def to_dict(self):
        return {"atom": self.atom, "role": self.role, "side": self.side,
                "sasa": round(self.sasa, 2), "label": self.label,
                "message": self.message}

    def __repr__(self):
        return "<Unsatisfied %s %s>" % (self.role, self.label)


def _sphere(n=SPHERE_POINTS):
    """Evenly spread unit vectors (golden-spiral)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


_UNIT = _sphere()


def accessible_area(ms, atoms, radii=None, present=None):
    """Probe-accessible surface of these atoms, in the whole system (A^2).

    Shrake-Rupley, for a handful of atoms at a time: points on each
    atom's expanded sphere, kept when no other atom's expanded sphere
    covers them. `present` leaves atoms out of the system - the ligand,
    to ask what a pocket atom could reach before anything was bound.
    """
    if radii is None:
        radii = np.array([vdw_radius(str(e)) for e in ms.element])
    total = 0.0
    for i in atoms:
        i = int(i)
        ri = radii[i] + PROBE
        d = ms.xyz - ms.xyz[i]
        near = np.flatnonzero((d * d).sum(axis=1)
                              < (ri + radii + PROBE) ** 2)
        near = near[near != i]
        if present is not None:
            near = near[present[near]]
        pts = ms.xyz[i] + ri * _UNIT
        free = np.ones(len(pts), dtype=bool)
        for j in near:
            dj = pts - ms.xyz[j]
            free &= (dj * dj).sum(axis=1) >= (radii[j] + PROBE) ** 2
            if not free.any():
                break
        total += 4.0 * np.pi * ri * ri * free.sum() / len(pts)
    return float(total)


def _share_a_neighbour(ms, i, j):
    return bool(set(int(x) for x in ms.neighbors(i))
                & set(int(x) for x in ms.neighbors(j)))


def find_unsatisfied(ms, feats, interactions):
    """Buried donors and acceptors, ligand and pocket, with no partner."""
    # Nitrogen and oxygen only. Perception also lists organic fluorine and
    # the like as weak acceptors, which is right for drawing a contact and
    # wrong here: nothing is owed to a C-F, and every fluorinated ligand
    # would carry a flag.
    strong = {i for i, e in enumerate(ms.element) if str(e) in ("N", "O")}
    donors = {f.root for f in feats.get("donors", ())} & strong
    acceptors = {f.root for f in feats.get("acceptors", ())
                 if f.moiety_id not in WEAK_ACCEPTORS} & strong
    polar = donors | acceptors
    if not polar:
        return []

    lig = np.asarray(ms.is_ligand, dtype=bool)
    water = np.asarray(ms.is_water, dtype=bool)
    hyd = np.asarray(ms.is_hydrogen, dtype=bool)
    lig_heavy = lig & ~hyd
    if not lig_heavy.any():
        return []

    # ---- who is judged: the ligand, and the pocket it sits against
    cand = {a for a in polar if lig[a]}
    lig_xyz = ms.xyz[lig_heavy]
    for a in polar:
        if lig[a] or water[a]:
            continue
        d = lig_xyz - ms.xyz[a]
        if (d * d).sum(axis=1).min() <= POCKET_RADIUS ** 2:
            cand.add(a)
    if not cand:
        return []

    # ---- who has a partner
    paired = set()
    for it in interactions or ():
        if it.kind in PAIRING_KINDS:
            paired.update(int(a) for a in it.lig_atoms)
            paired.update(int(a) for a in it.rec_atoms)

    has_h = bool(feats.get("has_hydrogens", True))
    dmask = np.zeros(ms.n_atoms, bool)
    amask = np.zeros(ms.n_atoms, bool)
    dmask[list(donors)] = True
    amask[list(acceptors)] = True
    ii, jj, dd = ms.pairs_within(CONTACT_MAX, dmask, amask)
    for d_i, a_i, dist in zip(ii, jj, dd):
        d_i, a_i = int(d_i), int(a_i)
        if d_i == a_i or (d_i not in cand and a_i not in cand):
            continue
        if a_i in set(int(x) for x in ms.neighbors(d_i)):
            continue
        if _share_a_neighbour(ms, d_i, a_i):
            continue                      # 1-3: geometry, not a choice
        hyds = ms.attached_hydrogens(d_i) if has_h else []
        if len(hyds):
            ok = any(ms.angle(d_i, int(h), a_i) >= DHA_MIN for h in hyds)
        else:
            ok = dist <= CONTACT_NO_H
        if ok:
            paired.add(d_i)
            paired.add(a_i)

    metal = np.array([str(e) in METALS for e in ms.element], dtype=bool)
    if metal.any():
        mxyz = ms.xyz[metal]
        for a in cand & acceptors:
            d = mxyz - ms.xyz[a]
            if (d * d).sum(axis=1).min() <= METAL_MAX ** 2:
                paired.add(a)

    # ---- of the rest, who is out of water's reach
    radii = np.array([vdw_radius(str(e)) for e in ms.element])
    out = []
    for a in sorted(cand - paired):
        group = [a] + [int(h) for h in ms.attached_hydrogens(a)]
        sasa = accessible_area(ms, group, radii)
        if sasa > BURIED_SASA:
            continue
        if not lig[a]:
            # A pocket atom is the pose's problem only if the pose is what
            # buried it. One that water could not reach in the empty site
            # either is the protein's own business - usually paired in a
            # way a truncated site or a missing hydrogen hides - and would
            # otherwise be flagged identically on every pose.
            if accessible_area(ms, group, radii, present=~lig) <= BURIED_SASA:
                continue
        role = ("donor/acceptor" if a in donors and a in acceptors
                else "donor" if a in donors else "acceptor")
        out.append(Unsatisfied(a, role, "ligand" if lig[a] else "receptor",
                               sasa, ms.atom_label(a)))
    return out

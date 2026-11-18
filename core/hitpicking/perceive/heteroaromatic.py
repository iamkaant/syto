"""Identify which heteroaromatic ring an aromatic atom belongs to.

Why this is worth doing
-----------------------
Hydrogen-bond basicity varies enormously across heteroaromatic nitrogens
that all look alike to coarse perception:

    pyridine      1.81 (median)      imidazole     2.57
    azines        1.22               other azoles  1.36
      pyrazine      0.92
      pyrimidine    1.07
      pyridazine    1.65

Calling all of them "a pyridine-type nitrogen" and showing every family at
once spans more than 1.5 log units, which is worse than useless: it is
authoritative-looking noise. A second ring nitrogen withdraws density and
drops the basicity sharply, and where in the ring it sits matters too.

Everything here is derived from ring composition and connectivity, so it
works identically on the ChimeraX path (no bond orders) and on parsed files.
"""

from __future__ import annotations

import numpy as np


def order_ring(ms, ring):
    """Ring atoms in connectivity order, starting from ring[0].

    Never assume the caller's order: our own ring finder returns a walked
    cycle, but ChimeraX's Ring.atoms is not guaranteed to be ordered, and
    positional chemistry (1,2 vs 1,3 vs 1,4) depends on getting this right.
    Returns None if the atoms do not form a single cycle.
    """
    members = set(int(a) for a in ring)
    if len(members) != len(ring):
        return None
    start = int(ring[0])
    order = [start]
    prev = None
    cur = start
    for _ in range(len(members) - 1):
        nxt = None
        for nb in ms.neighbors(cur):
            nb = int(nb)
            if nb in members and nb != prev and nb not in order:
                nxt = nb
                break
        if nxt is None:
            return None
        order.append(nxt)
        prev, cur = cur, nxt
    # the cycle must close
    if start not in [int(x) for x in ms.neighbors(cur)]:
        return None
    return order


def ring_separation(order, a, b):
    """Shortest number of bonds between two atoms around the ring."""
    n = len(order)
    ia, ib = order.index(a), order.index(b)
    d = abs(ia - ib)
    return min(d, n - d)


def smallest_aromatic_ring(ms, i):
    """The smallest aromatic ring containing atom i, or None."""
    best = None
    for ring, is_ar in zip(ms.rings(), ms.ring_aromatic()):
        if not is_ar or i not in ring:
            continue
        if best is None or len(ring) < len(best):
            best = ring
    return best


def fused_atoms(ms, ring):
    """Atoms of `ring` shared with another aromatic ring."""
    members = set(int(a) for a in ring)
    shared = set()
    for other, is_ar in zip(ms.rings(), ms.ring_aromatic()):
        if not is_ar:
            continue
        o = set(int(a) for a in other)
        if o == members:
            continue
        common = members & o
        if len(common) >= 2:
            shared |= common
    return shared


def classify_aromatic_nitrogen(ms, i):
    """Return a specific moiety id for an aromatic, non-protonated N.

    Falls back to the generic 'pyridine_n' only when the ring cannot be
    resolved, so a caller always gets something usable.
    """
    ring = smallest_aromatic_ring(ms, i)
    if ring is None:
        return "pyridine_n"
    order = order_ring(ms, ring)
    if order is None:
        return "pyridine_n"

    el = ms.element
    size = len(order)
    # ring heteroatoms, split by whether they carry a hydrogen
    n_noh, n_h, o_atoms, s_atoms = [], [], [], []
    for a in order:
        e = str(el[a])
        if e == "N":
            (n_h if len(ms.attached_hydrogens(a)) else n_noh).append(a)
        elif e == "O":
            o_atoms.append(a)
        elif e == "S":
            s_atoms.append(a)
    n_total = len(n_noh) + len(n_h)

    if size == 6:
        return _classify_six(ms, ring, order, i, n_total)
    if size == 5:
        return _classify_five(order, i, n_noh, n_h, o_atoms, s_atoms, n_total)
    return "pyridine_n"


def _classify_six(ms, ring, order, i, n_total):
    fused = fused_atoms(ms, ring)

    if n_total <= 1:
        if not fused:
            return "pyridine_n"
        # quinoline has its nitrogen next to a ring-fusion atom;
        # isoquinoline has it one position further round.
        neighbours_in_ring = {order[(order.index(i) + 1) % len(order)],
                              order[(order.index(i) - 1) % len(order)]}
        return "quinoline_n" if neighbours_in_ring & fused else "isoquinoline_n"

    if n_total == 2:
        others = [a for a in order
                  if a != i and str(ms.element[a]) == "N"]
        if not others:
            return "pyridine_n"
        sep = ring_separation(order, i, others[0])
        if sep == 1:
            # 1,2-diazine; fused to a benzo ring it is a phthalazine
            return "phthalazine_n" if fused else "pyridazine_n"
        if sep == 2:
            return "pyrimidine_n"
        return "pyrazine_n"              # sep == 3, para

    if n_total == 3:
        return "triazine_n"
    return "azine_n_polyaza"


def _classify_five(order, i, n_noh, n_h, o_atoms, s_atoms, n_total):
    if o_atoms:
        # oxazole: N and O 1,3   isoxazole: N and O adjacent
        return ("isoxazole_n2" if ring_separation(order, i, o_atoms[0]) == 1
                else "oxazole_n3")
    if s_atoms:
        return ("isothiazole_n2" if ring_separation(order, i, s_atoms[0]) == 1
                else "thiazole_n3")
    if n_total == 2:
        other = [a for a in (n_noh + n_h) if a != i]
        if not other:
            return "imidazole_n3"
        # pyrazole has its two nitrogens adjacent; imidazole has them 1,3
        return ("pyrazole_n2" if ring_separation(order, i, other[0]) == 1
                else "imidazole_n3")
    if n_total == 3:
        return "triazole_n"
    if n_total >= 4:
        return "tetrazole_n"
    return "azole_n"


def classify_aromatic_ring(ms, ring):
    """A descriptive id for a whole aromatic ring, used for stacking."""
    order = order_ring(ms, ring)
    atoms = order or [int(a) for a in ring]
    hetero = sorted({str(ms.element[a]) for a in atoms} - {"C"})
    if not hetero:
        return "aromatic_ring_6" if len(atoms) == 6 else "aromatic_ring_5"
    return "heteroaromatic_ring"

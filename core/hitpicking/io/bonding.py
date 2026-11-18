"""Infer covalent bonds from coordinates.

PDB files carry no bonds (CONECT is unreliable and usually absent for the
protein), so they are inferred from interatomic distance against covalent
radii. Metal coordination is deliberately NOT bonded: treating a Zn-N contact
as a covalent bond would corrupt hybridisation, ring perception and every
downstream H-bond count.
"""

from __future__ import annotations

import numpy as np

from ..elements import COVALENT_RADII, METALS, covalent_radius

# Cordero radii sum plus this slack counts as bonded. 0.45 A is the usual
# choice (it is what most viewers use) and separates 1.5 A bonds from the
# ~2.2 A shortest non-bonded contacts.
BOND_TOLERANCE = 0.45
MIN_BOND_LENGTH = 0.40


def infer_bonds(ms, tolerance: float = BOND_TOLERANCE, skip_metals: bool = True):
    """Return an (M, 2) int32 bond array inferred from distances."""
    el = ms.element
    radii = np.array([covalent_radius(str(e)) for e in el], dtype=np.float64)
    max_r = float(radii.max()) if len(radii) else 1.0
    cutoff = 2.0 * max_r + tolerance

    i, j, d = ms.pairs_within(cutoff)
    if len(i) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    limit = radii[i] + radii[j] + tolerance
    keep = (d < limit) & (d > MIN_BOND_LENGTH)

    isH = el == "H"
    keep &= ~(isH[i] & isH[j])          # no H-H bonds

    if skip_metals:
        is_metal = np.array([str(e) in METALS for e in el])
        keep &= ~(is_metal[i] | is_metal[j])

    i, j = i[keep], j[keep]
    bonds = np.stack([i, j], axis=1).astype(np.int32)

    # A hydrogen may only bond once: keep its shortest partner.
    bonds = _enforce_h_valence(ms, bonds)
    return bonds


def _enforce_h_valence(ms, bonds):
    if len(bonds) == 0:
        return bonds
    isH = ms.element == "H"
    hmask = isH[bonds[:, 0]] | isH[bonds[:, 1]]
    if not hmask.any():
        return bonds
    d = np.linalg.norm(ms.xyz[bonds[:, 0]] - ms.xyz[bonds[:, 1]], axis=1)
    best = {}
    drop = set()
    for bi in np.flatnonzero(hmask):
        a, b = int(bonds[bi, 0]), int(bonds[bi, 1])
        h = a if isH[a] else b
        prev = best.get(h)
        if prev is None:
            best[h] = bi
        elif d[bi] < d[prev]:
            drop.add(prev)
            best[h] = bi
        else:
            drop.add(bi)
    if not drop:
        return bonds
    keep = np.ones(len(bonds), dtype=bool)
    keep[list(drop)] = False
    return bonds[keep]

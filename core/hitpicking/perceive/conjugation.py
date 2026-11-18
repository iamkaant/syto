"""Lone-pair conjugation perception.

Why this exists
---------------
An amide nitrogen and an amine nitrogen have the same element, the same
degree and the same formal charge, and neither carries a double bond. They
are chemically miles apart: the amide lone pair is delocalised into the
carbonyl, which makes the nitrogen planar and a *very* poor hydrogen-bond
acceptor, while an aliphatic amine is one of the strongest acceptors a
drug-like molecule has (pKBHX ~2.5 vs essentially nil).

Getting this wrong would make the handbook wrong in the single place users
will look most. So conjugation is perceived explicitly, from connectivity and
geometry only, and never from bond orders -- ChimeraX cannot supply those.

The same pass runs on every input path (mol2, SDF, PDB, ChimeraX), so all
front ends agree.
"""

from __future__ import annotations

import numpy as np

from ..molsys import HYB_SP, HYB_SP2, HYB_SP3

CONJUGABLE = ("N", "O", "S")


def mark_conjugated(ms, set_hyb: bool = True, own_multiple_bond=None):
    """Flag heteroatoms whose lone pair is conjugated; optionally retype them.

    Returns a boolean array, also cached on ``ms.meta['conjugated']``.

    An N/O/S atom is treated as conjugated when it bears no multiple bond of
    its own but is bonded to an sp/sp2 or aromatic heavy atom -- the classic
    amide, ester, aniline, enamine and vinyl-ether situations. Sulfonamide and
    sulfone neighbours count too: the S there is formally sp3 but the N lone
    pair is still delocalised onto the oxygens.
    """
    n = ms.n_atoms
    conj = np.zeros(n, dtype=bool)
    retype = np.zeros(n, dtype=bool)
    el = ms.element

    # `own_multiple_bond[i]` says atom i is sp2/sp because of a double or
    # triple bond of its OWN, not because its lone pair is delocalised. Each
    # reader supplies it from what its format actually knows (Sybyl types,
    # MDL bond orders, residue templates). Without it we can only assume that
    # any already-sp2 atom earned that typing honestly, which would silently
    # lose every amide nitrogen -- the one distinction the handbook most needs.
    omb = (np.zeros(n, dtype=bool) if own_multiple_bond is None
           else np.asarray(own_multiple_bond, dtype=bool))
    assume_own = own_multiple_bond is None

    for i in range(n):
        e = str(el[i])
        if e not in CONJUGABLE:
            continue
        if ms.aromatic[i]:
            conj[i] = True          # aromatic heteroatoms are conjugated
            continue
        if omb[i]:
            continue                # genuinely multiply bonded; nothing to infer
        if assume_own and ms.hyb[i] in (HYB_SP, HYB_SP2):
            continue
        for j in ms.heavy_neighbors(i):
            j = int(j)
            if ms.aromatic[j] or ms.hyb[j] in (HYB_SP, HYB_SP2):
                conj[i] = True
                # Delocalisation into a pi system flattens the centre.
                retype[i] = True
                break
            if str(el[j]) == "S" and _is_sulfonyl(ms, j):
                # Sulfonamide / sulfonate: the lone pair is delocalised onto
                # the S=O oxygens, so the atom is a weak acceptor -- but the
                # centre stays pyramidal, so hybridisation is NOT changed.
                conj[i] = True
                break

    if set_hyb:
        # Never retype sulfur: a sulfonyl / sulfoxide S is tetrahedral even
        # though every neighbour looks sp2.
        retype &= (ms.hyb == HYB_SP3) & (el != "S")
        ms.hyb[retype] = HYB_SP2

    ms.meta["conjugated"] = conj
    return conj


def _is_sulfonyl(ms, s_idx: int) -> bool:
    """S bonded to two or more oxygens (sulfone / sulfonamide / sulfonate)."""
    nb = ms.heavy_neighbors(s_idx)
    return int((ms.element[nb] == "O").sum()) >= 2


def is_amide_nitrogen(ms, i: int) -> bool:
    """N single-bonded to a carbon that carries a double-bonded O or S.

    Detected without bond orders: the carbonyl carbon is sp2 and has an
    oxygen neighbour at a short, terminal distance.
    """
    if str(ms.element[i]) != "N":
        return False
    for c in ms.heavy_neighbors(i):
        c = int(c)
        if str(ms.element[c]) != "C" or ms.hyb[c] != HYB_SP2:
            continue
        for o in ms.heavy_neighbors(c):
            o = int(o)
            if o == i:
                continue
            if str(ms.element[o]) in ("O", "S") and len(ms.heavy_neighbors(o)) == 1:
                return True
    return False


def is_carbonyl_oxygen(ms, i: int) -> bool:
    """Terminal O on an sp2 carbon."""
    if str(ms.element[i]) != "O":
        return False
    hv = ms.heavy_neighbors(i)
    if len(hv) != 1 or len(ms.attached_hydrogens(i)) > 0:
        return False
    c = int(hv[0])
    return str(ms.element[c]) == "C" and ms.hyb[c] == HYB_SP2


def is_ether_oxygen(ms, i: int) -> bool:
    """O with two heavy neighbours and no hydrogen (ether or ester -O-)."""
    if str(ms.element[i]) != "O":
        return False
    return len(ms.heavy_neighbors(i)) == 2 and len(ms.attached_hydrogens(i)) == 0

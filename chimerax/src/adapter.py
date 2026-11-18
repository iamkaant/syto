"""ChimeraX AtomicStructure  <->  hitpicking.MolSystem.

The whole point of the shared core is that this file is the ONLY place that
knows about ChimeraX. Everything downstream - perception, detection, scoring,
the handbook - is the same code the web page runs.

ChimeraX stores no bond orders. It does store IDATM types, which carry more
chemistry than bond orders do for our purposes: `Npl` is a planar nitrogen
whose lone pair is delocalised (amide, aniline), while `N2` is a nitrogen with
a genuine double bond of its own. That distinction is exactly what the
conjugation pass needs, and it is why nothing here has to guess.
"""

from __future__ import annotations

import numpy as np

from hitpicking.molsys import (MolSystem, HYB_SP, HYB_SP2, HYB_SP3,
                               HYB_UNKNOWN, PART_RECEPTOR, PART_LIGAND,
                               PART_WATER, PART_ION, PART_COFACTOR)
from hitpicking.perceive.conjugation import mark_conjugated
from hitpicking import aa_templates as T

# IDATM geometry codes -> hybridisation.
#   1 = single, 2 = linear, 3 = planar, 4 = tetrahedral
_GEOM_TO_HYB = {1: HYB_SP3, 2: HYB_SP, 3: HYB_SP2, 4: HYB_SP3}

# Aromatic IDATM types.
AROMATIC_IDATM = frozenset({"Car", "Nar", "Oar", "Oar+", "Sar"})

# IDATM types whose planarity/linearity comes from a multiple bond the atom
# genuinely owns, as opposed to a delocalised lone pair. Everything NOT in
# here that still looks sp2 is a conjugation candidate.
OWN_MULTIPLE_BOND_IDATM = frozenset({
    "C1", "C1-", "C2", "Cac",
    "N1", "N1+", "N2", "N2+", "Ntr",
    "O1", "O1+", "O2", "O2-",
    "P ac", "Pac", "Pox",
    "S2", "Sac", "Son", "Sxd",
})

# IDATM's trailing '+' and '-' mark a charge-CAPABLE type, not an actual
# formal charge. ChimeraX types neutral trimethylamine as N3+ just as it does
# a protonated amine, so reading the '+' as +1 marked every tertiary amine
# cationic - which then became a phantom "ammonium" group and, through it, a
# phantom salt bridge.
#
# The charge is therefore derived from connectivity, which is unambiguous.


def _formal_charge(idatm, element, n_connections, n_hydrogens):
    if element == "N":
        # four substituents on nitrogen is a quaternary or protonated amine
        if n_connections >= 4:
            return 1
        # sp2 nitrogen bearing three substituents: an iminium. A NEUTRAL
        # three-coordinate sp2 nitrogen is typed Npl (amide, aniline), not N2+.
        if idatm == "N2+" and n_connections == 3:
            return 1
        if idatm == "N1+" and n_connections == 2:
            return 1
        # Ng+ is a guanidinium nitrogen: the charge belongs to the whole
        # delocalised group, so perceive.features assigns it there.
        return 0
    if element == "O":
        # O2-/O3- are carboxylate, phosphate and nitro oxygens, where the
        # charge is shared. Putting -1 on each would double-count it, so the
        # group perceiver owns these too.
        return 0
    if element == "S" and idatm == "S3-" and n_connections == 1 \
            and n_hydrogens == 0:
        return -1          # thiolate
    return 0


def _hyb_from_idatm(idatm, element):
    try:
        from chimerax.atomic import idatm as idatm_mod
        info = idatm_mod.type_info.get(idatm)
    except Exception:
        info = None
    if info is not None:
        return _GEOM_TO_HYB.get(getattr(info, "geometry", 4), HYB_UNKNOWN)
    if element == "H":
        return HYB_SP3
    return HYB_UNKNOWN


def _part_for_residue(residue, ligand_residues=None, is_ligand_structure=False):
    name = (residue.name or "").strip().upper()
    if is_ligand_structure:
        return PART_LIGAND
    if ligand_residues and residue in ligand_residues:
        return PART_LIGAND
    if name in T.WATER_RESNAMES:
        return PART_WATER
    if name in T.ION_RESNAMES:
        return PART_ION
    if name in T.STANDARD_AA:
        return PART_RECEPTOR
    return PART_COFACTOR


def from_structures(structures, ligand_structures=(), ligand_residues=None):
    """Build one MolSystem from one or more ChimeraX structures.

    `ligand_structures` are marked PART_LIGAND wholesale (the ViewDock case,
    where each pose is its own model). `ligand_residues` marks individual
    residues instead, for a ligand that lives inside the receptor model.

    Returns (molsys, atom_list) where atom_list[i] is the ChimeraX Atom for
    index i, so results can be mapped back to the viewer.
    """
    if not isinstance(structures, (list, tuple)):
        structures = [structures]
    lig_set = set(ligand_structures or ())

    all_atoms = []
    radii = []
    xyz, element, name, hyb, arom, fchg = [], [], [], [], [], []
    part, res_name, res_seq, chain_id = [], [], [], []
    own_mb = []
    offset = {}
    n = 0

    for s in structures:
        atoms = s.atoms
        offset[s] = n
        coords = atoms.scene_coords          # scene coords: models may be moved
        at_radii = atoms.radii               # ChimeraX's own vdW radii
        els = atoms.element_names
        idatms = atoms.idatm_types
        names = atoms.names
        residues = atoms.residues
        is_lig_struct = s in lig_set

        for k in range(len(atoms)):
            a = atoms[k]
            all_atoms.append(a)
            el = str(els[k])
            it = str(idatms[k]) if idatms[k] else ""
            xyz.append(coords[k])
            radii.append(float(at_radii[k]))
            element.append(el)
            name.append(str(names[k])[:6])
            hyb.append(_hyb_from_idatm(it, el))
            arom.append(it in AROMATIC_IDATM)
            a_conn = len(a.neighbors)
            a_h = sum(1 for nb in a.neighbors if nb.element.name == "H")
            fchg.append(_formal_charge(it, el, a_conn, a_h))
            own_mb.append(it in OWN_MULTIPLE_BOND_IDATM)
            r = residues[k]
            res_name.append((r.name or "")[:4])
            res_seq.append(int(r.number))
            chain_id.append((r.chain_id or "")[:2])
            part.append(_part_for_residue(r, ligand_residues, is_lig_struct))
        n += len(atoms)

    # bonds, remapped into the merged index space
    bond_list = []
    for s in structures:
        b = s.bonds
        if len(b) == 0:
            continue
        a1, a2 = b.atoms
        base = offset[s]
        i1 = s.atoms.indices(a1)
        i2 = s.atoms.indices(a2)
        for p, q in zip(i1, i2):
            if p < 0 or q < 0:
                continue
            bond_list.append((base + int(p), base + int(q)))

    ms = MolSystem(
        np.asarray(xyz, dtype=np.float64),
        np.asarray(element, dtype="<U2"),
        np.asarray(bond_list, dtype=np.int32).reshape(-1, 2),
        name=np.asarray(name, dtype="<U6"),
        hyb=np.asarray(hyb, dtype=np.int8),
        aromatic=np.asarray(arom, dtype=bool),
        formal_charge=np.asarray(fchg, dtype=np.int8),
        part=np.asarray(part, dtype=np.int8),
        res_name=np.asarray(res_name, dtype="<U4"),
        res_seq=np.asarray(res_seq, dtype=np.int32),
        chain_id=np.asarray(chain_id, dtype="<U2"),
        meta={"source": "chimerax", "format": "chimerax",
              # Pass ChimeraX's per-atom radii through, so clash detection
              # here agrees with the Clashes tool rather than quietly using
              # a different radius table.
              "vdw_radii": np.asarray(radii, dtype=np.float64)},
    )

    # ChimeraX's own ring perception is authoritative; trust it over ours.
    _apply_chimerax_rings(ms, structures, offset)

    repaired = repair_disulfides(ms)
    if repaired:
        ms.meta["repaired_disulfides"] = repaired

    ms.meta["has_hydrogens"] = bool((ms.element == "H").any())
    mark_conjugated(ms, set_hyb=True,
                    own_multiple_bond=np.asarray(own_mb, dtype=bool))
    return ms, all_atoms


# Disulfide S-S bond length is ~2.05 A; anything under this and not already
# bonded is a disulfide the file failed to declare.
SS_BOND_MAX = 2.30


def repair_disulfides(ms):
    """Add disulfide bonds that the source connectivity is missing.

    A prepared receptor often has no SSBOND records, and ChimeraX will then
    leave the two cysteines unbonded. That is not cosmetic: an unbonded SG
    with no hydrogen is perceived as nothing at all, whereas a disulfide
    sulfur is a weak acceptor, so a missing bond silently changes the
    chemistry of the site. Deliberately narrow - only S-S, only within
    covalent range - so it cannot invent other bonds.

    Returns a list of (i, j) pairs added.
    """
    s_mask = ms.element == "S"
    if int(s_mask.sum()) < 2:
        return []
    ii, jj, dd = ms.pairs_within(SS_BOND_MAX, s_mask, s_mask)
    if len(ii) == 0:
        return []
    existing = set()
    for a, b in ms.bonds:
        existing.add((int(a), int(b)))
        existing.add((int(b), int(a)))
    added = []
    for a, b, d in zip(ii, jj, dd):
        a, b = int(a), int(b)
        if a >= b or (a, b) in existing:
            continue
        if ms.residue_key(a) == ms.residue_key(b):
            continue
        added.append((a, b))
    if added:
        ms.bonds = np.concatenate(
            [ms.bonds, np.asarray(added, dtype=np.int32)]).astype(np.int32)
        ms._adj = None
        ms._rings = None
    return added


def _apply_chimerax_rings(ms, structures, offset):
    """Use ChimeraX's ring perception for aromaticity where it is available."""
    rings, arom_flags = [], []
    for s in structures:
        base = offset[s]
        try:
            found = s.rings()
        except Exception:
            continue
        for r in found:
            try:
                idx = s.atoms.indices(r.atoms)
            except Exception:
                continue
            if len(idx) == 0 or min(idx) < 0:
                continue
            ring = tuple(base + int(i) for i in idx)
            rings.append(ring)
            is_ar = bool(getattr(r, "aromatic", False))
            arom_flags.append(is_ar)
            if is_ar:
                for a in ring:
                    ms.aromatic[a] = True
    if rings:
        order = sorted(range(len(rings)), key=lambda i: len(rings[i]))
        ms._rings = [rings[i] for i in order]
        ms._ring_aromatic = [arom_flags[i] for i in order]


def atoms_to_indices(all_atoms):
    """{ChimeraX Atom: index} for mapping a pick back into the MolSystem."""
    return {a: i for i, a in enumerate(all_atoms)}


def indices_to_atoms(all_atoms, indices):
    """ChimeraX Atoms collection for a list of MolSystem indices."""
    from chimerax.atomic import Atoms
    sel = [all_atoms[int(i)] for i in indices
           if 0 <= int(i) < len(all_atoms)]
    return Atoms(sel)


def find_ligand_structures(session, exclude=()):
    """Heuristic: small, non-polymer structures are the docked ligands.

    ViewDock opens each pose as its own model, which is the normal case; this
    is the fallback for a user who has just opened files by hand.
    """
    from chimerax.atomic import AtomicStructure
    out = []
    for m in session.models:
        if not isinstance(m, AtomicStructure) or m in exclude:
            continue
        try:
            if m.num_residues <= 3 and m.num_atoms <= 200:
                out.append(m)
        except Exception:
            continue
    return out


def find_receptor_structure(session):
    """The largest polymer structure in the session."""
    from chimerax.atomic import AtomicStructure
    best, best_n = None, 0
    for m in session.models:
        if not isinstance(m, AtomicStructure):
            continue
        try:
            n = m.num_residues
        except Exception:
            continue
        if n > best_n:
            best, best_n = m, n
    return best if best_n > 10 else None

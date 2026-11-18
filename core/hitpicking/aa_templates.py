"""Standard residue templates: hybridisation, aromaticity and charged groups.

Protein chemistry is taken from templates rather than perceived from geometry.
A docking receptor is almost always standard residues, and templates get
histidine tautomers, amide nitrogens and carboxylates right where a geometric
guess would not.

Charged groups are named, not reduced to per-atom integers, because salt
bridges are measured between *group* centroids (both carboxylate oxygens,
all three guanidinium nitrogens).
"""

from __future__ import annotations

from .molsys import HYB_SP, HYB_SP2, HYB_SP3

# Atom names that are aromatic ring members, per residue.
AROMATIC_ATOMS = {
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HID": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HIE": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HIP": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
}

# Non-aromatic sp2 atoms, per residue (side chain only; backbone handled below).
SP2_ATOMS = {
    "ARG": {"NE", "CZ", "NH1", "NH2"},
    "ASN": {"CG", "OD1", "ND2"},
    "ASP": {"CG", "OD1", "OD2"},
    "GLN": {"CD", "OE1", "NE2"},
    "GLU": {"CD", "OE1", "OE2"},
    "TYR": set(),      # OH is sp3
}

# Backbone: N is a planar amide nitrogen, C=O is sp2.
BACKBONE_SP2 = {"N", "C", "O", "OXT"}

# Charged side-chain groups: residue -> (group_type, atom names, net charge).
# Protonation is the single biggest source of error here; see `notes`.
CHARGED_GROUPS = {
    "ASP": ("carboxylate", ("OD1", "OD2", "CG"), -1),
    "GLU": ("carboxylate", ("OE1", "OE2", "CD"), -1),
    "ARG": ("guanidinium", ("NE", "CZ", "NH1", "NH2"), +1),
    "LYS": ("ammonium", ("NZ",), +1),
    "HIP": ("imidazolium", ("ND1", "CE1", "NE2", "CG", "CD2"), +1),
}

# Residues whose charge state is assumed and should be surfaced to the user.
ASSUMED_PROTONATION = {
    "ASP": "assumed deprotonated (-1)",
    "GLU": "assumed deprotonated (-1)",
    "LYS": "assumed protonated (+1)",
    "ARG": "assumed protonated (+1)",
    "HIS": "neutral; tautomer not determined from coordinates alone",
    "CYS": "assumed neutral thiol (not thiolate)",
    "TYR": "assumed neutral phenol",
}

STANDARD_AA = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "MSE", "SEC", "PYL",
})

WATER_RESNAMES = frozenset({"HOH", "WAT", "H2O", "TIP", "TIP3", "SOL", "DOD"})

# Monatomic ions commonly present in receptors.
ION_RESNAMES = frozenset({
    "ZN", "MG", "CA", "NA", "K", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI",
    "CD", "HG", "CL", "BR", "IOD", "F", "SO4", "PO4",
})

# Residues the hit-picking wiki calls out as flexible: their side chains can
# reorient to make or break an interaction, so a missing contact here is less
# damning than elsewhere. ("Think of receptor flexibility, especially His, Gln,
# Asn, Ser, Tyr, Thr" - wiki.docking.org/Hit_picking_party)
FLEXIBLE_SIDECHAINS = frozenset({"HIS", "HID", "HIE", "HIP", "GLN", "ASN",
                                 "SER", "TYR", "THR"})


def classify_atom(res_name: str, atom_name: str, element: str):
    """Return (hyb, aromatic) for a standard-residue atom.

    Unknown residues fall back to (HYB_UNKNOWN=0, False) so the caller can
    perceive geometrically instead.
    """
    rn = (res_name or "").strip().upper()
    an = (atom_name or "").strip().upper()

    if element == "H":
        return HYB_SP3, False
    if rn in WATER_RESNAMES:
        return HYB_SP3, False
    if rn in ION_RESNAMES and len(an) <= 2:
        return HYB_SP3, False

    if rn in AROMATIC_ATOMS and an in AROMATIC_ATOMS[rn]:
        return HYB_SP2, True
    if an in BACKBONE_SP2:
        return HYB_SP2, False
    if rn in SP2_ATOMS and an in SP2_ATOMS[rn]:
        return HYB_SP2, False
    if rn in STANDARD_AA:
        return HYB_SP3, False
    return 0, False


def charged_group_for(res_name: str):
    """(group_type, atom_names, net_charge) or None."""
    return CHARGED_GROUPS.get((res_name or "").strip().upper())

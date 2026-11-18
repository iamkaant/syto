"""Element reference data.

Every table here cites its source. Values are NOT to be edited without
updating the citation. Where two conventions exist (e.g. hydrogen vdW
radius) the alternatives are given explicitly rather than silently chosen.
"""

from __future__ import annotations

# Covalent radii, Angstrom.
# Cordero et al., Dalton Trans. 2008, 2832-2838. DOI: 10.1039/b801115j
# Transition metals: low-spin values used where the paper gives a choice.
COVALENT_RADII = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Pb": 1.46,
}
# Carbon is hybridisation dependent (same source, Table 2).
COVALENT_RADII_C = {3: 0.76, 2: 0.73, 1: 0.69}

# van der Waals radii, Angstrom.
# Bondi, J. Phys. Chem. 1964, 68, 441-451. DOI: 10.1021/j100785a001
# Extended for main-group elements Bondi did not cover by
# Mantina et al., J. Phys. Chem. A 2009, 113, 5806-5812. DOI: 10.1021/jp8111556
VDW_RADII = {
    "H": 1.20, "He": 1.40,
    "Li": 1.81, "Be": 1.53, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52,
    "F": 1.47, "Ne": 1.54,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80,
    "Cl": 1.75, "Ar": 1.88,
    "K": 2.75, "Ca": 2.31, "Ni": 1.63, "Cu": 1.40, "Zn": 1.39,
    "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02,
    "Rb": 3.03, "Sr": 2.49, "Pd": 1.63, "Ag": 1.72, "Cd": 1.58,
    "In": 1.93, "Sn": 2.17, "Sb": 2.06, "Te": 2.06, "I": 1.98, "Xe": 2.16,
    "Pt": 1.75, "Au": 1.66, "Hg": 1.55, "Pb": 2.02,
    # Transition metals absent from both sources; Cordero covalent + 0.8 is a
    # crude stand-in. Flagged so clash code can treat them as low confidence.
    "Fe": 2.05, "Mn": 2.05, "Co": 2.00, "Mo": 2.10,
}
# Elements whose vdW radius above is an estimate, not a literature value.
VDW_ESTIMATED = frozenset({"Fe", "Mn", "Co", "Mo"})

# ChimeraX / AMBER use a smaller hydrogen radius than Bondi. The clash code
# takes a radius table as an argument so the two conventions can be compared;
# find_clashes parity tests need this one.
VDW_RADII_AMBER_H = dict(VDW_RADII, H=1.00)

# Pauling electronegativity.
# Allred, J. Inorg. Nucl. Chem. 1961, 17, 215-221. DOI: 10.1016/0022-1902(61)80142-5
ELECTRONEGATIVITY = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04,
    "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96,
    "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66,
}

HALOGENS = frozenset({"F", "Cl", "Br", "I"})
# Halogens able to form a sigma-hole halogen bond. F is excluded: it is too
# electronegative and too small to develop a positive sigma-hole in practice.
# Bissantz, Kuhn & Stahl, J. Med. Chem. 2010, 53, 5061-5084. DOI: 10.1021/jm100112j
SIGMA_HOLE_HALOGENS = frozenset({"Cl", "Br", "I"})

CHALCOGENS = frozenset({"O", "S", "Se"})

METALS = frozenset({
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Pt", "Au", "Hg", "Pb", "Al", "Ga", "In", "Sn",
})

# Elements that can carry a lone pair relevant to H-bond acceptance.
POLAR_ACCEPTOR_ELEMENTS = frozenset({"N", "O", "S", "Se", "F", "Cl", "Br", "I"})
# Elements that, bearing a hydrogen, act as H-bond donors.
POLAR_DONOR_ELEMENTS = frozenset({"N", "O", "S"})


def covalent_radius(element: str, hyb: int = 0) -> float:
    """Covalent radius in Angstrom. Falls back to 0.77 (generic C) if unknown."""
    if element == "C" and hyb in COVALENT_RADII_C:
        return COVALENT_RADII_C[hyb]
    return COVALENT_RADII.get(element, 0.77)


def vdw_radius(element: str, table: dict = None) -> float:
    """van der Waals radius in Angstrom. Falls back to 1.70 (C) if unknown."""
    t = VDW_RADII if table is None else table
    return t.get(element, 1.70)


def normalize_element(raw: str) -> str:
    """Normalise an element token from a file to 'Cl' / 'C' capitalisation.

    Handles PDB-style right-justified 2-char fields, Sybyl atom types that
    carry a subtype ('C.ar', 'N.pl3'), and names with trailing digits ('CA1').
    """
    if not raw:
        return ""
    s = raw.strip()
    if "." in s:            # Sybyl mol2 atom type
        s = s.split(".", 1)[0]
    # strip digits and charge marks
    s = "".join(ch for ch in s if ch.isalpha())
    if not s:
        return ""
    if len(s) == 1:
        return s.upper()
    cand = s[0].upper() + s[1].lower()
    if cand in COVALENT_RADII:
        return cand
    # Two-letter token that is not an element: probably a PDB atom name
    # like 'CA' (alpha carbon) or 'NE2'. Fall back to the first letter.
    first = s[0].upper()
    if first in COVALENT_RADII:
        return first
    return cand

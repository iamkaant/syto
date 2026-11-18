"""Pharmacophoric feature perception: donors, acceptors, charged groups, rings.

This is the layer the handbook attaches to. An atom is not simply "an
oxygen": it is a carbonyl oxygen, an ether oxygen or a carboxylate oxygen,
and those three differ by orders of magnitude in hydrogen-bond basicity.
Feature classes here are therefore named after the moiety, and each carries
the ``moiety_id`` that links it to a handbook entry.

Works with or without explicit hydrogens. Many docking receptors (and some
ligand files) have none; in that case donors are inferred from heavy-atom
valence and the caller is told that H-dependent geometry is unavailable.
"""

from __future__ import annotations

import numpy as np

from ..molsys import HYB_SP, HYB_SP2, HYB_SP3
from .. import aa_templates as T
from ..elements import SIGMA_HOLE_HALOGENS

# ---------------------------------------------------------------------------
# Acceptor classes, ordered strongest-first within each element.
# `moiety_id` keys into the handbook; `strength_hint` is a coarse ordering used
# only for display grouping, never as an energy.
# ---------------------------------------------------------------------------

ACCEPTOR_NONE = None

# Short descriptions for the specific heteroaromatic nitrogens.
_AROMATIC_N_NOTES = {
    "pyridine_n": "pyridine-type N; strong acceptor",
    "quinoline_n": "quinoline N; strong acceptor",
    "isoquinoline_n": "isoquinoline N; strong acceptor",
    "pyridazine_n": "1,2-diazine N; weakened by the adjacent N",
    "pyrimidine_n": "1,3-diazine N; markedly weaker than pyridine",
    "pyrazine_n": "1,4-diazine N; among the weakest azine acceptors",
    "phthalazine_n": "phthalazine N",
    "triazine_n": "triazine N; strongly deactivated",
    "azine_n_polyaza": "polyaza ring N; strongly deactivated",
    "imidazole_n3": "imidazole N3; the strongest common azole acceptor",
    "pyrazole_n2": "pyrazole N2",
    "oxazole_n3": "oxazole N3",
    "isoxazole_n2": "isoxazole N2; weakened by the adjacent O",
    "thiazole_n3": "thiazole N3",
    "isothiazole_n2": "isothiazole N2",
    "triazole_n": "triazole N",
    "tetrazole_n": "tetrazole N",
    "azole_n": "azole N",
}



# ---------------------------------------------------------------------------
# The registry of every moiety id this module can emit.
#
# This is the contract between perception and the handbook: tools/build_data.py
# rejects any handbook entry naming an id that is not here, and a test asserts
# perception never emits one that is not here either. Add to both sides
# deliberately, never by accident.
# ---------------------------------------------------------------------------

AROMATIC_N_IDS = frozenset(_AROMATIC_N_NOTES)

ACCEPTOR_IDS = frozenset({
    "carboxylate_o", "phosphate_sulfonate_o", "amide_o", "carbonyl_o",
    "sulfoxide_o", "sulfone_o", "n_oxide_o", "nitro_o", "phosphine_oxide_o",
    "phenol_oh", "alcohol_oh", "furan_o", "ester_o", "ether_o", "water_o",
    "generic_o", "pyridine_n", "nitrile_n", "imine_n", "amine_n_tertiary",
    "amine_n_secondary", "amine_n_primary", "aniline_n", "thiocarbonyl_s",
    "thioether_s", "thiol_s", "organic_f",
}) | AROMATIC_N_IDS

DONOR_IDS = frozenset({
    "ammonium_n", "charged_n_donor", "pyrrole_nh", "amide_nh", "amine_nh",
    "carboxylic_oh", "water_oh", "phenol_oh_donor", "alcohol_oh_donor",
    "thiol_sh",
})

GROUP_IDS = frozenset({
    "carboxylate", "guanidinium", "ammonium", "imidazolium", "amidinium",
    "sulfonate", "phosphate",
})

RING_IDS = frozenset({"aromatic_ring_6", "aromatic_ring_5", "heteroaromatic_ring"})

HALOGEN_IDS = frozenset({"halogen_cl", "halogen_br", "halogen_i"})

OTHER_IDS = frozenset({"hydrophobic", "sulfonamide_n",
                       "pyrrole_type_n_substituted", "amide_n_tertiary"})

MOIETY_IDS = (ACCEPTOR_IDS | DONOR_IDS | GROUP_IDS | RING_IDS
              | HALOGEN_IDS | OTHER_IDS)




class Feature:
    """One perceived feature: a moiety instance on specific atoms."""

    __slots__ = ("kind", "moiety_id", "atoms", "root", "note")

    def __init__(self, kind, moiety_id, atoms, root=None, note=""):
        self.kind = kind                  # 'donor' | 'acceptor' | 'cation' | ...
        self.moiety_id = moiety_id        # handbook key
        self.atoms = tuple(int(a) for a in atoms)
        self.root = int(root) if root is not None else self.atoms[0]
        self.note = note

    def __repr__(self):
        return "<%s %s atoms=%s>" % (self.kind, self.moiety_id, self.atoms)

    def to_dict(self):
        return {"kind": self.kind, "moiety_id": self.moiety_id,
                "atoms": list(self.atoms), "root": self.root, "note": self.note}


# ---------------------------------------------------------------- acceptors

def classify_acceptor(ms, i: int):
    """Return (moiety_id, note) if atom i accepts a hydrogen bond, else None.

    The negative cases matter as much as the positive ones: amide and
    sulfonamide nitrogens, pyrrole-type NH, and anilines are *not* acceptors,
    and saying so is part of the tool's job.
    """
    el = str(ms.element[i])
    if el not in ("N", "O", "S", "F"):
        return None

    hv = ms.heavy_neighbors(i)
    nh = len(ms.attached_hydrogens(i))
    conj = ms.meta.get("conjugated")
    is_conj = bool(conj[i]) if conj is not None else False
    chg = int(ms.formal_charge[i])

    if el == "O":
        if chg < 0 or _in_carboxylate(ms, i):
            return "carboxylate_o", "anionic; strong acceptor"
        if _is_phosphate_or_sulfonate_o(ms, i):
            return "phosphate_sulfonate_o", "anionic oxygen"
        if len(hv) == 1 and nh == 0 and ms.hyb[i] == HYB_SP2:
            c = int(hv[0])
            if str(ms.element[c]) == "C":
                if _c_bears_n(ms, c, exclude=i):
                    return "amide_o", "amide carbonyl; good acceptor"
                return "carbonyl_o", "ketone/aldehyde carbonyl"
            if str(ms.element[c]) == "S":
                return "sulfoxide_o" if _n_oxygens(ms, c) == 1 else "sulfone_o", ""
            if str(ms.element[c]) == "N":
                # A nitro group has TWO terminal oxygens on the nitrogen; an
                # N-oxide has one. They are chemically opposite: the nitro
                # oxygens are very poor acceptors, the N-oxide is a strong one.
                if len(_terminal_oxygens(ms, c)) >= 2:
                    return "nitro_o", "nitro oxygen; very weak acceptor"
                return "n_oxide_o", "N-oxide; strong acceptor"
            if str(ms.element[c]) == "P":
                return "phosphine_oxide_o", "very strong acceptor"
        if nh >= 1 and len(hv) == 1:
            return ("phenol_oh", "aromatic hydroxyl") if ms.aromatic[int(hv[0])] \
                else ("alcohol_oh", "aliphatic hydroxyl")
        if len(hv) == 2 and nh == 0:
            if ms.aromatic[i]:
                return "furan_o", "aromatic ether; weak acceptor"
            if _is_ester_o(ms, i):
                return "ester_o", "ester -O-; weak acceptor"
            return "ether_o", "ether oxygen"
        if nh == 2 and len(hv) == 0:
            return "water_o", "water oxygen"
        return "generic_o", ""

    if el == "N":
        if chg > 0:
            return None                   # protonated N donates, never accepts
        if ms.aromatic[i] and (nh > 0 or len(hv) >= 3):
            # Pyrrole-type ring nitrogen: its lone pair is part of the
            # aromatic sextet, so it does not accept - whether it carries a
            # hydrogen (pyrrole, imidazole N1-H) or a substituent
            # (N-methylimidazole N1). Only the other, two-coordinate ring
            # nitrogen of an azole is an acceptor.
            return None
        if is_conj and _is_amide_like(ms, i):
            return None                   # amide / sulfonamide N: not an acceptor
        if ms.aromatic[i] and nh == 0 and len(hv) == 2:
            # Which heteroaromatic ring this is matters: pyridine, pyrazine
            # and imidazole nitrogens differ by more than 1.5 log units of
            # basicity, so reporting them all as "pyridine-type" is noise.
            from .heteroaromatic import classify_aromatic_nitrogen
            mid = classify_aromatic_nitrogen(ms, i)
            return mid, _AROMATIC_N_NOTES.get(mid, "aromatic N; acceptor")
        if ms.hyb[i] == HYB_SP and len(hv) == 1:
            return "nitrile_n", "nitrile; moderate, highly directional"
        if ms.hyb[i] == HYB_SP2 and nh == 0 and len(hv) == 2:
            return "imine_n", "imine/amidine N"
        if ms.hyb[i] == HYB_SP3 and not is_conj:
            # An sp3 nitrogen with a full complement of substituents plus a
            # hydrogen is already protonated: it donates, it does not accept.
            if nh > 0 and (len(hv) + nh) >= 4:
                return None
            n_sub = len(hv)
            if n_sub == 3:
                return "amine_n_tertiary", "tertiary amine; usually protonated"
            if n_sub == 2:
                return "amine_n_secondary", "secondary amine"
            return "amine_n_primary", "primary amine"
        if is_conj and nh == 0:
            return "aniline_n", "conjugated N; weak acceptor"
        return None

    if el == "S":
        if len(hv) == 1 and nh == 0 and ms.hyb[i] == HYB_SP2:
            return "thiocarbonyl_s", "C=S; moderate acceptor"
        if len(hv) == 2 and nh == 0:
            return "thioether_s", "weak acceptor"
        if nh >= 1:
            return "thiol_s", "weak"
        return None

    if el == "F":
        # Organic fluorine is a very poor acceptor; reported so users stop
        # counting it as a hydrogen bond.
        if len(hv) == 1 and str(ms.element[int(hv[0])]) == "C":
            return "organic_f", "very weak; rarely a real H-bond"
    return None


def _is_amide_like(ms, i):
    """N attached to C=O, C=S, S(=O)2 or a nitro group."""
    for j in ms.heavy_neighbors(i):
        j = int(j)
        ej = str(ms.element[j])
        if ej == "C" and ms.hyb[j] == HYB_SP2:
            for k in ms.heavy_neighbors(j):
                k = int(k)
                if k != i and str(ms.element[k]) in ("O", "S") \
                        and len(ms.heavy_neighbors(k)) == 1 \
                        and len(ms.attached_hydrogens(k)) == 0:
                    return True
        if ej == "S" and _n_oxygens(ms, j) >= 2:
            return True
    return False


def _c_bears_n(ms, c, exclude=-1):
    for k in ms.heavy_neighbors(c):
        if int(k) != exclude and str(ms.element[int(k)]) == "N":
            return True
    return False


def _n_oxygens(ms, i):
    return int(sum(1 for j in ms.heavy_neighbors(i) if str(ms.element[int(j)]) == "O"))


def _terminal_oxygens(ms, c):
    """Oxygens on c that are terminal and carry no hydrogen."""
    out = []
    for j in ms.heavy_neighbors(c):
        j = int(j)
        if str(ms.element[j]) != "O":
            continue
        if len(ms.heavy_neighbors(j)) == 1 and len(ms.attached_hydrogens(j)) == 0:
            out.append(j)
    return out


def _in_carboxylate(ms, i):
    """O belonging to a -C(=O)O(-) group with both oxygens terminal."""
    rn = str(ms.res_name[i]).strip().upper()
    if rn in ("ASP", "GLU") and str(ms.name[i]).strip().upper() in (
            "OD1", "OD2", "OE1", "OE2"):
        return True
    hv = ms.heavy_neighbors(i)
    if len(hv) != 1:
        return False
    c = int(hv[0])
    if str(ms.element[c]) != "C":
        return False
    return len(_terminal_oxygens(ms, c)) == 2


def _is_phosphate_or_sulfonate_o(ms, i):
    hv = ms.heavy_neighbors(i)
    if len(hv) != 1:
        return False
    x = int(hv[0])
    return str(ms.element[x]) in ("P", "S") and _n_oxygens(ms, x) >= 3


def _is_ester_o(ms, i):
    """-O- flanked by a carbonyl carbon."""
    for j in ms.heavy_neighbors(i):
        j = int(j)
        if str(ms.element[j]) == "C" and ms.hyb[j] == HYB_SP2 \
                and _terminal_oxygens(ms, j):
            return True
    return False


# ------------------------------------------------------------------ donors

def classify_non_acceptor(ms, i):
    """Atoms commonly mistaken for acceptors. Returns (moiety_id, why) or None.

    Half of what gets argued about at a hit-picking party is someone counting
    one of these, so the handbook says so explicitly rather than just staying
    silent about the atom.
    """
    el = str(ms.element[i])
    if el != "N":
        return None
    hv = ms.heavy_neighbors(i)
    nh = len(ms.attached_hydrogens(i))
    if ms.aromatic[i] and (nh > 0 or len(hv) >= 3):
        if nh > 0:
            return ("pyrrole_nh",
                    "pyrrole-type NH: donates, does not accept")
        return ("pyrrole_type_n_substituted",
                "substituted pyrrole-type ring N: lone pair is in the "
                "aromatic sextet, so it neither donates nor accepts")
    if _is_amide_like(ms, i) and nh == 0:
        return ("amide_n_tertiary",
                "tertiary amide N: delocalised into the carbonyl, not an "
                "acceptor")
    return None


def classify_donor(ms, i: int):
    """Return (moiety_id, note) if atom i is a hydrogen-bond donor heavy atom."""
    el = str(ms.element[i])
    if el not in ("N", "O", "S"):
        return None
    nh = len(ms.attached_hydrogens(i))
    has_h_info = bool(ms.meta.get("has_hydrogens", True))

    if nh == 0 and has_h_info:
        # Explicit hydrogens are present and this atom has none.
        if int(ms.formal_charge[i]) > 0 and el == "N":
            return ("ammonium_n", "cationic N (H positions not in file)")
        return None
    if nh == 0 and not has_h_info:
        # No hydrogens anywhere: infer from valence.
        if not _could_bear_hydrogen(ms, i):
            return None

    chg = int(ms.formal_charge[i])
    if el == "N":
        if chg > 0 or _is_guanidinium_n(ms, i) or _is_amidinium_n(ms, i):
            return "charged_n_donor", "cationic donor; strong"
        if ms.aromatic[i]:
            return "pyrrole_nh", "aromatic NH"
        if _is_amide_like(ms, i):
            return "amide_nh", "amide NH"
        return "amine_nh", "amine NH"
    if el == "O":
        if _in_carboxylate(ms, i):
            return "carboxylic_oh", "COOH (if protonated)"
        if len(ms.heavy_neighbors(i)) == 0:
            return "water_oh", "water donor"
        if len(ms.heavy_neighbors(i)) == 1 and ms.aromatic[int(ms.heavy_neighbors(i)[0])]:
            return "phenol_oh_donor", "phenol OH"
        return "alcohol_oh_donor", "hydroxyl"
    if el == "S":
        return "thiol_sh", "weak donor"
    return None


def _could_bear_hydrogen(ms, i):
    """Valence-based donor inference for files with no explicit hydrogens."""
    el = str(ms.element[i])
    nheavy = len(ms.heavy_neighbors(i))
    if el == "N":
        return nheavy <= 2 or (nheavy == 3 and ms.hyb[i] == HYB_SP3)
    if el == "O":
        return nheavy <= 1
    if el == "S":
        return nheavy <= 1
    return False


def _is_guanidinium_n(ms, i):
    for c in ms.heavy_neighbors(i):
        c = int(c)
        if str(ms.element[c]) != "C" or ms.hyb[c] != HYB_SP2:
            continue
        ns = [int(k) for k in ms.heavy_neighbors(c) if str(ms.element[int(k)]) == "N"]
        if len(ns) >= 3:
            return True
    return False


def _is_amidinium_n(ms, i):
    for c in ms.heavy_neighbors(i):
        c = int(c)
        if str(ms.element[c]) != "C" or ms.hyb[c] != HYB_SP2:
            continue
        ns = [int(k) for k in ms.heavy_neighbors(c) if str(ms.element[int(k)]) == "N"]
        if len(ns) == 2 and all(len(ms.heavy_neighbors(x)) <= 2 for x in ns):
            # exclude amides: a true amidine has no terminal carbonyl O
            if not _terminal_oxygens(ms, c):
                return True
    return False


# ---------------------------------------------------------- charged groups

def _prot_note(ms, what):
    if ms.meta.get("has_hydrogens", True):
        return "%s, protonated in the input file" % what
    return ("%s; no hydrogens in the input, so the charge state is ASSUMED "
            "(+1 at pH 7) - check the pKa" % what)


def _group_is_protonated(ms, n_atoms_list, need_all=True):
    """Decide whether a delocalised base is actually protonated.

    When the file carries explicit hydrogens it is the authority: in a
    guanidinium or amidinium the proton is delocalised, so *every* nitrogen
    of the group bears at least one H. A 2-aminopyridine whose ring N has no
    hydrogen is the neutral base, and its ring N is an acceptor, not a cation.

    With no hydrogens anywhere in the file we cannot tell, so these groups are
    assumed protonated (the usual medicinal-chemistry default) and the caller
    marks them ambiguous.
    """
    if any(int(ms.formal_charge[a]) > 0 for a in n_atoms_list):
        return True
    if not ms.meta.get("has_hydrogens", True):
        return True                       # no evidence either way
    counts = [len(ms.attached_hydrogens(a)) for a in n_atoms_list]
    return all(c >= 1 for c in counts) if need_all else any(c >= 1 for c in counts)


def find_charged_groups(ms):
    """Perceive ionisable groups as *groups*, not per-atom integers.

    Salt bridges are measured between group centroids (both carboxylate
    oxygens, all three guanidinium nitrogens), so the group is the unit.
    Returns a list of Feature with kind 'cation' or 'anion'.
    """
    out = []
    seen = set()
    n = ms.n_atoms

    # Protein side chains come from templates: more reliable than perception.
    # Residues are indexed once; scanning all atoms per residue was quadratic
    # and dominated the runtime on a 3k-atom receptor.
    by_residue = {}
    for j in range(n):
        by_residue.setdefault(ms.residue_key(j), []).append(j)

    for key, members in by_residue.items():
        rn = str(key[2]).strip().upper()
        grp = T.charged_group_for(rn)
        if not grp:
            continue
        gtype, atom_names, charge = grp
        atoms = [j for j in members
                 if str(ms.name[j]).strip().upper() in atom_names]
        if not atoms:
            continue
        gkey = (key, gtype)
        if gkey in seen:
            continue
        seen.add(gkey)
        out.append(Feature("cation" if charge > 0 else "anion",
                           gtype, atoms, root=atoms[0],
                           note=T.ASSUMED_PROTONATION.get(rn, "")))

    # Ligand (and anything without a template): perceive from structure.
    for i in range(n):
        if str(ms.res_name[i]).strip().upper() in T.CHARGED_GROUPS:
            continue
        el = str(ms.element[i])
        if el == "C" and ms.hyb[i] == HYB_SP2:
            ns = [int(k) for k in ms.heavy_neighbors(i) if str(ms.element[int(k)]) == "N"]
            if len(ns) >= 3 and not _terminal_oxygens(ms, i):
                key = ("guan", i)
                if key not in seen and _group_is_protonated(ms, ns):
                    seen.add(key)
                    out.append(Feature("cation", "guanidinium", [i] + ns, root=i,
                                       note=_prot_note(ms, "guanidine")))
            elif len(ns) == 2 and not _terminal_oxygens(ms, i) \
                    and all(len(ms.heavy_neighbors(x)) <= 2 for x in ns):
                key = ("amid", i)
                if key not in seen and _group_is_protonated(ms, ns):
                    seen.add(key)
                    out.append(Feature("cation", "amidinium", [i] + ns, root=i,
                                       note=_prot_note(ms, "amidine")))
        elif el == "C" and ms.hyb[i] == HYB_SP2:
            pass
        elif el in ("C",):
            pass

        if el == "C":
            tos = _terminal_oxygens(ms, i)
            if ms.hyb[i] == HYB_SP2 and len(tos) == 2:
                key = ("carbox", i)
                if key not in seen:
                    seen.add(key)
                    out.append(Feature("anion", "carboxylate", [i] + tos, root=i,
                                       note="assumed deprotonated at pH 7"))
        if el in ("S", "P"):
            tos = _terminal_oxygens(ms, i)
            if len(tos) >= 3:
                key = ("oxyanion", i)
                if key not in seen:
                    seen.add(key)
                    out.append(Feature(
                        "anion", "sulfonate" if el == "S" else "phosphate",
                        [i] + tos, root=i, note="assumed deprotonated"))
        if el == "N" and int(ms.formal_charge[i]) > 0:
            key = ("ncat", i)
            if key not in seen:
                seen.add(key)
                out.append(Feature("cation", "ammonium", [i], root=i,
                                   note="formal +1 from input file"))
        elif el == "N" and ms.hyb[i] == HYB_SP3 and not ms.aromatic[i]:
            conj = ms.meta.get("conjugated")
            if (conj is None or not conj[i]) and not _is_amide_like(ms, i) \
                    and str(ms.res_name[i]).strip().upper() not in T.STANDARD_AA:
                key = ("amine_cat", i)
                protonated = (not ms.meta.get("has_hydrogens", True)) or \
                    (len(ms.attached_hydrogens(i)) + len(ms.heavy_neighbors(i)) >= 4)
                if key not in seen and protonated:
                    seen.add(key)
                    out.append(Feature("cation", "ammonium", [i], root=i,
                                       note=_prot_note(ms, "aliphatic amine")))
    return out


# ------------------------------------------------------------------- rings

def find_aromatic_rings(ms, ligand_only=False):
    """Aromatic rings as Features, with centroid/normal available on demand."""
    out = []
    for ring, is_ar in zip(ms.rings(), ms.ring_aromatic()):
        if not is_ar:
            continue
        if ligand_only and not all(ms.is_ligand[a] for a in ring):
            continue
        hetero = sorted({str(ms.element[a]) for a in ring} - {"C"})
        mid = "aromatic_ring_6" if len(ring) == 6 else "aromatic_ring_5"
        if hetero:
            mid = "heteroaromatic_ring"
        out.append(Feature("ring", mid, ring, root=ring[0],
                           note="+".join(hetero) if hetero else "carbocyclic"))
    return out


def find_halogen_donors(ms):
    """Cl/Br/I on carbon: capable of a sigma-hole halogen bond."""
    out = []
    for i in range(ms.n_atoms):
        el = str(ms.element[i])
        if el not in SIGMA_HOLE_HALOGENS:
            continue
        hv = ms.heavy_neighbors(i)
        if len(hv) == 1:
            out.append(Feature("halogen", "halogen_%s" % el.lower(), [i],
                               root=i, note="sigma-hole donor"))
    return out


def find_hydrophobic_atoms(ms):
    """Carbon and sulfur with no polar neighbour: the greasy surface."""
    mask = np.zeros(ms.n_atoms, dtype=bool)
    for i in range(ms.n_atoms):
        el = str(ms.element[i])
        if el not in ("C", "S", "F", "Cl", "Br", "I"):
            continue
        if el == "C":
            polar = any(str(ms.element[int(j)]) in ("N", "O")
                        for j in ms.heavy_neighbors(i))
            if polar:
                continue
        mask[i] = True
    return mask


# --------------------------------------------------------------- top level

def perceive_all(ms, note_missing_hydrogens=True):
    """Run every perceiver. Returns a dict of feature lists.

    Also records whether the system has explicit hydrogens: without them,
    H-dependent geometry (the D-H...A angle) cannot be evaluated and the
    detectors fall back to heavy-atom criteria.
    """
    has_h = bool((ms.element == "H").any())
    ms.meta["has_hydrogens"] = has_h

    donors, acceptors = [], []
    for i in range(ms.n_atoms):
        if str(ms.element[i]) == "H":
            continue
        d = classify_donor(ms, i)
        if d:
            donors.append(Feature("donor", d[0], [i], root=i, note=d[1]))
        a = classify_acceptor(ms, i)
        if a:
            acceptors.append(Feature("acceptor", a[0], [i], root=i, note=a[1]))

    out = {
        "donors": donors,
        "acceptors": acceptors,
        "charged": find_charged_groups(ms),
        "rings": find_aromatic_rings(ms),
        "halogens": find_halogen_donors(ms),
        "hydrophobic": find_hydrophobic_atoms(ms),
        "has_hydrogens": has_h,
    }
    if note_missing_hydrogens and not has_h:
        out["warnings"] = [
            "No explicit hydrogens in this structure. Donor assignment is "
            "inferred from heavy-atom valence and D-H...A angles cannot be "
            "evaluated; H-bond geometry scores will be distance-only."
        ]
    else:
        out["warnings"] = []
    return out

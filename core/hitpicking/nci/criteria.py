"""Geometric criteria and energy ranges for non-covalent interactions.

STATUS OF THE NUMBERS IN THIS FILE
----------------------------------
The *geometric* criteria below are standard, long-established values and are
cited individually.

The *energy* ranges are coarse, order-of-magnitude estimates. They are marked
``needs_review`` and are shown to users as ranges with an explicit caveat.
They are NOT free energies, they are not additive, and they must not be summed
into a score. They exist so the tool can say "this looks like a good H-bond"
rather than "this is worth -1.7 kcal/mol".

These defaults are superseded, entry by entry, once ``data-src/`` is curated
from the literature (see docs/LITERATURE_EXTRACTION_PROMPT.md). Anything still
carrying ``curation_status='needs_review'`` renders with a warning marker in
both front ends.
"""

from __future__ import annotations

NEEDS_REVIEW = "needs_review"
VERIFIED = "verified"


class Criterion:
    """Geometric envelope for one interaction type.

    Distances in Angstrom, angles in degrees. ``d_ideal`` scores 1.0;
    the score falls to 0 at ``d_min``/``d_max``.
    """

    __slots__ = ("key", "label", "d_min", "d_ideal", "d_max", "angles",
                 "energy", "refs", "curation_status", "notes",
                 "distance_measure")

    def __init__(self, key, label, d_min, d_ideal, d_max, angles=None,
                 energy=None, refs=(), curation_status=NEEDS_REVIEW, notes="",
                 distance_measure="atoms"):
        # Which distance the limits above refer to. Getting this wrong is not
        # a detail: for parallel stacking the literature 3.4-3.8 A is the
        # separation between the ring PLANES, while for T-shaped it is the
        # centroid-to-centroid distance, and the two diverge as soon as the
        # rings are laterally displaced.
        self.distance_measure = distance_measure
        self.key = key
        self.label = label
        self.d_min = d_min
        self.d_ideal = d_ideal
        self.d_max = d_max
        # name -> (min_acceptable, ideal). Score falls to 0 at min.
        self.angles = dict(angles or {})
        # (low, typical, high) kcal/mol, magnitude of a favourable contribution
        self.energy = energy
        self.refs = tuple(refs)
        self.curation_status = curation_status
        self.notes = notes

    def to_dict(self):
        return {"key": self.key, "label": self.label,
                "distance_measure": self.distance_measure,
                "d_min": self.d_min, "d_ideal": self.d_ideal, "d_max": self.d_max,
                "angles": self.angles, "energy": self.energy,
                "refs": list(self.refs), "curation_status": self.curation_status,
                "notes": self.notes}


# --------------------------------------------------------------------------
# Hydrogen bonds
#
# Heavy-atom D...A range and the D-H...A angle cutoff follow the classical
# surveys of protein and small-molecule structures.
#   Baker & Hubbard, Prog. Biophys. Mol. Biol. 1984, 44, 97-179.
#     DOI: 10.1016/0079-6107(84)90007-5
#   Mills & Dean, J. Comput. Aided Mol. Des. 1996, 10, 607-622.
#     DOI: 10.1007/BF00134183   (the geometry ChimeraX's hbonds also uses)
#   Bissantz, Kuhn & Stahl, J. Med. Chem. 2010, 53, 5061-5084.
#     DOI: 10.1021/jm100112j
# --------------------------------------------------------------------------

HBOND = Criterion(
    "hbond", "Hydrogen bond",
    d_min=2.40, d_ideal=2.90, d_max=3.50,
    angles={
        # D-H...A: linear is ideal; below 120 deg it is not a hydrogen bond.
        "dha": (120.0, 180.0),
        # H...A-X: the acceptor's lone pair must point roughly at the H.
        "haa": (90.0, 135.0),
    },
    energy=(0.5, 1.5, 2.5),
    refs=("10.1016/0079-6107(84)90007-5", "10.1007/BF00134183",
          "10.1021/jm100112j"),
    notes=("Energy is the NET contribution to binding in water, which is much "
           "smaller than the gas-phase interaction because donor and acceptor "
           "were both solvated beforehand. A buried, well-formed H-bond that "
           "replaces nothing is worth more than a solvent-exposed one."),
)

# Charge-assisted H-bond (one partner ionised). Shorter and stronger.
HBOND_CHARGED = Criterion(
    "hbond_charged", "Charge-assisted hydrogen bond",
    d_min=2.30, d_ideal=2.75, d_max=3.40,
    angles={"dha": (120.0, 180.0), "haa": (90.0, 135.0)},
    energy=(1.0, 2.5, 4.0),
    refs=("10.1021/jm100112j",),
    notes="Ionised donor or acceptor; shorter and stronger than neutral.",
)

WEAK_HBOND = Criterion(
    "weak_hbond", "Weak hydrogen bond (C-H...O / X-H...pi)",
    d_min=2.80, d_ideal=3.40, d_max=3.90,
    angles={"dha": (110.0, 180.0)},
    energy=(0.1, 0.4, 0.8),
    refs=("10.1021/jm100112j",),
    notes=("Individually negligible; occasionally matters in aggregate. Shown "
           "only when explicitly enabled."),
)

# --------------------------------------------------------------------------
# Salt bridges
#   Barlow & Thornton, J. Mol. Biol. 1983, 168, 867-885.
#     DOI: 10.1016/S0022-2836(83)80079-5   (the 4 A N/O criterion)
# --------------------------------------------------------------------------

SALT_BRIDGE = Criterion(
    "salt_bridge", "Salt bridge",
    d_min=2.50, d_ideal=3.20, d_max=5.00,
    angles={},
    energy=(1.0, 2.5, 5.0),
    refs=("10.1016/S0022-2836(83)80079-5", "10.1021/jm100112j"),
    notes=("Measured between the closest charged atoms of the two groups. "
           "A SOLVENT-EXPOSED salt bridge contributes far less than a buried "
           "one; desolvating a charge is expensive and can cancel the gain "
           "entirely. Treat a surface salt bridge with scepticism."),
)

# --------------------------------------------------------------------------
# Aromatic stacking
#   McGaughey, Gelb & Rose, J. Biol. Chem. 1998, 273, 15458-15463.
#     DOI: 10.1074/jbc.273.25.15458
#   Bissantz 2010 (geometry of the parallel-displaced vs T-shaped minima)
# --------------------------------------------------------------------------

PI_STACK_PARALLEL = Criterion(
    "pi_stack_parallel", "pi-stacking (parallel-displaced)",
    # NOTE these are the PERPENDICULAR separation of the two ring planes, not
    # the centroid-to-centroid distance. Scoring the centroid distance here
    # accepted rings 5 A apart as stacked (because a 5 A centroid separation
    # sits inside a 5.5 A window) while rejecting a perfectly good 3.4 A stack
    # that happened to be laterally displaced by 3 A.
    d_min=2.90, d_ideal=3.50, d_max=4.20,
    distance_measure="interplanar_separation",
    angles={
        # angle between the two ring planes; 0 = coplanar
        "interplanar": (30.0, 0.0),
    },
    energy=(0.5, 1.5, 3.0),
    refs=("10.1074/jbc.273.25.15458", "10.1021/jm100112j"),
    notes=("Perfectly face-to-face (sandwich) stacking is NOT the minimum: "
           "rings prefer a lateral offset of roughly 1.5 A, which is scored "
           "separately. The reported distance is the plane separation; the "
           "centroid distance is given alongside it."),
)
# Preferred lateral offset window (A) for parallel stacking. Parallel-displaced
# geometries reach ~3 A of offset while still stacking, so the window runs
# wider than the ~1.5 A optimum.
PI_STACK_OFFSET = (0.0, 1.5, 3.5)
# A zero-offset sandwich is a real stacking geometry, just a less favourable
# one than parallel-displaced, so it scores here rather than at either
# extreme of full marks or rejection.
PI_STACK_OFFSET_FLOOR = 0.45

PI_STACK_TSHAPED = Criterion(
    "pi_stack_tshaped", "pi-stacking (T-shaped / edge-to-face)",
    # Here the centroid-to-centroid distance IS the right measure: the rings
    # are perpendicular, so there is no meaningful plane separation.
    d_min=4.00, d_ideal=5.00, d_max=6.00,
    distance_measure="centroid",
    angles={"interplanar": (60.0, 90.0)},
    energy=(0.5, 1.2, 2.5),
    refs=("10.1074/jbc.273.25.15458", "10.1021/jm100112j"),
)

# --------------------------------------------------------------------------
# Cation-pi
#   Gallivan & Dougherty, Proc. Natl. Acad. Sci. 1999, 96, 9459-9464.
#     DOI: 10.1073/pnas.96.17.9459
# --------------------------------------------------------------------------

CATION_PI = Criterion(
    "cation_pi", "Cation-pi",
    # cation to ring centroid
    d_min=3.00, d_ideal=4.30, d_max=6.00,
    distance_measure="centroid",
    angles={
        # angle between the ring normal and the centroid->cation vector;
        # 0 deg means the cation sits directly over the ring face.
        "normal_offset": (45.0, 0.0),
    },
    energy=(1.0, 2.5, 5.0),
    refs=("10.1073/pnas.96.17.9459", "10.1021/jm100112j"),
    notes="Strongest when the cation sits over the ring face, not its edge.",
)

# --------------------------------------------------------------------------
# Halogen bonds
#   Auffinger, Hays, Westhof & Ho, Proc. Natl. Acad. Sci. 2004, 101, 16789.
#     DOI: 10.1073/pnas.0407607101
#   Bissantz 2010 (sigma-hole directionality)
# --------------------------------------------------------------------------

HALOGEN_BOND = Criterion(
    "halogen_bond", "Halogen bond",
    d_min=2.70, d_ideal=3.20, d_max=4.00,
    angles={
        # C-X...A must be near-linear: the sigma hole lies on the C-X axis.
        "cxa": (140.0, 180.0),
        # X...A-Y: approach towards the acceptor lone pair.
        "xay": (90.0, 120.0),
    },
    energy=(0.5, 1.5, 3.0),
    refs=("10.1073/pnas.0407607101", "10.1021/jm100112j"),
    notes=("Strength increases I > Br > Cl. Fluorine does NOT form halogen "
           "bonds: it has no sigma hole."),
)

# --------------------------------------------------------------------------
# Hydrophobic contact
# --------------------------------------------------------------------------

HYDROPHOBIC = Criterion(
    "hydrophobic", "Hydrophobic contact",
    d_min=3.20, d_ideal=3.90, d_max=4.50,
    angles={},
    energy=(0.0, 0.2, 0.4),
    refs=("10.1021/jm100112j",),
    notes=("Per-atom-pair value is tiny; burying a large apolar surface is "
           "what actually pays. Counted, not summed into an energy."),
)

ALL = {
    c.key: c for c in (
        HBOND, HBOND_CHARGED, WEAK_HBOND, SALT_BRIDGE,
        PI_STACK_PARALLEL, PI_STACK_TSHAPED, CATION_PI,
        HALOGEN_BOND, HYDROPHOBIC,
    )
}

# Types enabled by default in the UI. Weak H-bonds and hydrophobic contacts
# are off by default: they are numerous and drown out the real signal.
DEFAULT_ENABLED = ("hbond", "hbond_charged", "salt_bridge",
                   "pi_stack_parallel", "pi_stack_tshaped",
                   "cation_pi", "halogen_bond")

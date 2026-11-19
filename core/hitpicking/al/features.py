"""Feature vectors for the active-learning model.

AutoParty uses a LUNA interaction fingerprint folded to 4096 bits, plus (in
the user's fork) MACCS keys and a handful of RDKit descriptors. Neither LUNA
nor RDKit can run here: LUNA needs Python 3.7, openbabel and biopython 1.72,
and RDKit is absent from ChimeraX and from Pyodide's default set. So the
features are built from what this tool already computes.

Three blocks, each versioned together as FEATURE_VERSION. A model trained
under one version must not be loaded under another, because the columns mean
different things - hitpicking.al.store refuses to.

  1. an interaction fingerprint, hashed over
     (interaction type, ligand moiety, receptor residue) and folded;
  2. a ligand-structure fingerprint, a Morgan-like hash over atom
     environments, folded;
  3. a short descriptor block: counts, charges, clash and strain, and the
     docking program's own score terms where the pose carries them.

The interaction block is what makes this an INTERACTION model rather than a
ligand model: two poses of the same molecule differ in it, which is the
whole point when grading poses rather than compounds.
"""

from __future__ import annotations

import numpy as np

from .. import FEATURE_VERSION

IFP_BITS = 1024
LIGAND_BITS = 1024
N_DESCRIPTORS = 21
TOTAL_BITS = IFP_BITS + LIGAND_BITS + N_DESCRIPTORS

# Score terms worth feeding in, where a DOCK pose carries them. These are
# the keys ViewDock puts on the structure, verified against DOCK mol2
# output rather than guessed.
DOCK_TERMS = ("Total Energy", "Van der Waals", "Electrostatic",
              "Ligand Polar Desolv", "Ligand Apolar Desolv",
              "Total Strain", "Max Strain")

# Where each block starts and ends. The learner weights these against one
# another: standardising per column leaves the ligand block with ~90% of
# the variance simply because it has ~765 varying columns to the
# interaction block's ~67, and equal per column is not equal per block.
BLOCKS = (("interactions", 0, IFP_BITS),
          ("ligand", IFP_BITS, IFP_BITS + LIGAND_BITS),
          ("descriptors", IFP_BITS + LIGAND_BITS, TOTAL_BITS))


def _fold(text, bits):
    """A stable hash into [0, bits). Python's hash() is salted per process,
    so it cannot be used: a model trained in one session would index
    different columns in the next."""
    import hashlib
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % bits


def interaction_fingerprint(result, bits=IFP_BITS, weighted=True):
    """Hash each detected interaction into a bit vector.

    Keyed on the interaction type, the ligand-side moiety and the receptor
    residue, so "a charge-assisted hydrogen bond from an amide NH to ASP149"
    is one feature. Counts are weighted by geometric quality when `weighted`,
    so a textbook contact and a barely-qualifying one are not equal.
    """
    v = np.zeros(bits, dtype=np.float64)
    ms = result.combined
    for it in result.interactions:
        lig_m = it.moiety_ids[0] if it.moiety_ids else "?"
        rec_atoms = it.rec_atoms or ()
        res = "?"
        if rec_atoms and ms is not None:
            i = int(rec_atoms[0])
            res = "%s%d" % (str(ms.res_name[i]).strip(), int(ms.res_seq[i]))
        v[_fold("%s|%s|%s" % (it.kind, lig_m, res), bits)] += (
            it.quality if weighted else 1.0)
        # a type-and-moiety feature too, so a pose that moves to a
        # neighbouring residue still shares signal with its siblings
        v[_fold("%s|%s" % (it.kind, lig_m), bits)] += (
            it.quality if weighted else 1.0)
    return v


def ligand_fingerprint(ms, bits=LIGAND_BITS, radius=2):
    """Morgan-like hash of ligand atom environments.

    Built on MolSystem rather than RDKit: element, hybridisation,
    aromaticity, charge and attached hydrogens, grown outward `radius`
    bonds.
    """
    v = np.zeros(bits, dtype=np.float64)
    lig = np.flatnonzero(ms.is_ligand & (ms.element != "H"))
    if len(lig) == 0:
        return v
    labels = {}
    for i in lig:
        i = int(i)
        labels[i] = "%s%d%s%+d h%d" % (
            str(ms.element[i]), int(ms.hyb[i]),
            "a" if ms.aromatic[i] else "", int(ms.formal_charge[i]),
            len(ms.attached_hydrogens(i)))
    for i in labels:
        v[_fold(labels[i], bits)] += 1.0
    for _ in range(radius):
        nxt = {}
        for i in labels:
            env = sorted(labels[int(j)] for j in ms.heavy_neighbors(i)
                         if int(j) in labels)
            nxt[i] = labels[i] + "(" + ",".join(env) + ")"
        labels = nxt
        for i in labels:
            v[_fold(labels[i], bits)] += 1.0
    return v


def descriptors(result):
    """A short block of per-pose numbers.

    The first thirteen are geometry and counts, scaled to roughly [0, 1].
    The rest are the docking program's own score terms, which the model
    was previously blind to: PoseResult used to be built with no scores at
    all, so the two slots reserved for them sat at zero for every pose.
    """
    ms = result.combined
    d = np.zeros(N_DESCRIPTORS, dtype=np.float64)
    if ms is None:
        return d
    lig = ms.is_ligand
    heavy = int((lig & (ms.element != "H")).sum())
    c = result.counts or {}

    d[0] = heavy / 50.0
    d[1] = (c.get("hbond", 0) + c.get("hbond_charged", 0)) / 10.0
    d[2] = c.get("salt_bridge", 0) / 4.0
    d[3] = (c.get("pi_stack_parallel", 0) + c.get("pi_stack_tshaped", 0)) / 4.0
    d[4] = c.get("cation_pi", 0) / 3.0
    d[5] = c.get("halogen_bond", 0) / 3.0
    d[6] = sum(1 for i in result.interactions if i.quality >= 0.55) / 8.0
    d[7] = float(np.sum(ms.formal_charge[lig])) / 3.0
    d[8] = int(ms.aromatic[lig].sum()) / 20.0

    strain = getattr(result, "strain", None)
    if strain is not None:
        sc = strain.counts
        d[9] = min(strain.worst_overlap, 2.0) / 2.0
        d[10] = sc.get("clashes", 0) / 5.0
        d[11] = sc.get("internal_clashes", 0) / 5.0
        d[12] = sc.get("flagged_torsions", 0) / 5.0

    d[13] = 1.0 if (ms.meta.get("has_hydrogens") or False) else 0.0

    # DOCK energies are negative and of order tens. The division only puts
    # them on a comparable footing; the per-column standardisation does the
    # real scaling. The clip is there for DOCK's sentinel values - a failed
    # term comes through as 999.999, which would otherwise swamp the column
    # mean and standard deviation for every pose in the set.
    scores = result.scores or {}
    for k, name in enumerate(DOCK_TERMS):
        val = scores.get(name)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            v = float(val)
            if v != v or abs(v) > 500.0:      # NaN, or a sentinel
                continue
            d[14 + k] = float(np.clip(v / 50.0, -4.0, 4.0))
    return d


def pose_features(result, weighted=True):
    """The full vector for one analysed pose."""
    ms = result.combined
    parts = [interaction_fingerprint(result, weighted=weighted),
             ligand_fingerprint(ms) if ms is not None
             else np.zeros(LIGAND_BITS),
             descriptors(result)]
    return np.concatenate(parts)


def feature_info():
    return {"version": FEATURE_VERSION, "total": TOTAL_BITS,
            "ifp_bits": IFP_BITS, "ligand_bits": LIGAND_BITS,
            "n_descriptors": N_DESCRIPTORS}

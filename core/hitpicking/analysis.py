"""Top-level analysis API. This is what the front ends call.

A hit-picking party steps through hundreds of poses in one binding site, so
the receptor is perceived ONCE and reused. Combined systems always place the
receptor first, which keeps receptor atom indices stable across every pose.

Typical use:

    ctx = ReceptorContext.from_pdb("rec.pdb")
    for pose in read_mol2_iter("poses.mol2"):
        result = ctx.analyze(pose)
        print(result.counts, result.to_dict())
"""

from __future__ import annotations

import numpy as np

from .molsys import MolSystem, PART_LIGAND
from .nci import criteria as C
from .nci.detect import detect_interactions
from .perceive.features import perceive_all


class Options:
    """User-facing toggles. Every feature of the tool is switchable."""

    __slots__ = ("enabled_nci", "include_weak_hbonds", "site_radius",
                 "show_lengths", "show_angles", "detect_strain",
                 "show_clashes", "show_strain_labels",
                 "handbook_on_click", "annotate_atom_values", "use_active_learning")

    def __init__(self, enabled_nci=None, include_weak_hbonds=False,
                 site_radius=10.0, show_lengths=True, show_angles=False,
                 detect_strain=True, show_clashes=True,
                 show_strain_labels=True, handbook_on_click=True,
                 annotate_atom_values=False, use_active_learning=False):
        self.enabled_nci = set(C.DEFAULT_ENABLED if enabled_nci is None
                               else enabled_nci)
        self.include_weak_hbonds = include_weak_hbonds
        # Receptor atoms beyond this distance from the ligand are ignored.
        # Purely a speed optimisation; it must never change the result, so it
        # is validated against the full receptor in the test suite.
        self.site_radius = site_radius
        self.show_lengths = show_lengths
        self.show_angles = show_angles
        self.detect_strain = detect_strain
        self.show_clashes = show_clashes
        self.show_strain_labels = show_strain_labels
        self.handbook_on_click = handbook_on_click
        self.annotate_atom_values = annotate_atom_values
        self.use_active_learning = use_active_learning

    def to_dict(self):
        return {k: (sorted(getattr(self, k)) if k == "enabled_nci"
                    else getattr(self, k)) for k in self.__slots__}


class PoseResult:
    """Everything known about one pose."""

    def __init__(self, name, interactions, warnings, counts, scores=None,
                 combined=None, n_receptor=0, strain=None):
        self.name = name
        self.strain = strain
        self.interactions = interactions
        self.warnings = warnings
        self.counts = counts
        self.scores = scores or {}
        self.combined = combined
        self.n_receptor = n_receptor

    def ligand_index(self, combined_index):
        """Map an index in the combined system back to the ligand's own."""
        return combined_index - self.n_receptor

    def to_dict(self):
        d = {
            "name": self.name,
            "counts": self.counts,
            "warnings": self.warnings,
            "scores": self.scores,
            "n_receptor": self.n_receptor,
            "interactions": [i.to_dict() for i in self.interactions],
        }
        if self.strain is not None:
            d["strain"] = self.strain.to_dict()
        return d

    def summary(self):
        parts = ["%s %d" % (k.replace("_", " "), v)
                 for k, v in sorted(self.counts.items())]
        out = "; ".join(parts) if parts else "no interactions detected"
        if self.strain is not None:
            c = self.strain.counts
            if c["clashes"] or c["internal_clashes"] or c["strain_flags"]:
                out += "  |  " + self.strain.summary()
        return out


class ReceptorContext:
    """A receptor, perceived once, ready to be paired with many poses."""

    def __init__(self, receptor: MolSystem, options: Options = None):
        self.receptor = receptor
        self.options = options or Options()
        self.features = perceive_all(receptor)
        self.has_hydrogens = self.features["has_hydrogens"]
        self.warnings = list(self.features.get("warnings", []))
        if not self.has_hydrogens:
            self.warnings.append(
                "The receptor has no explicit hydrogens. Run ChimeraX 'addh' "
                "(or equivalent) for meaningful hydrogen-bond geometry.")

    @classmethod
    def from_pdb(cls, path, options=None, **kw):
        from .io.pdb import read_pdb
        return cls(read_pdb(path, **kw), options)

    # -------------------------------------------------------------- combine

    def combine(self, ligand: MolSystem):
        """Receptor first, ligand appended: receptor indices never move."""
        rec = self.receptor
        n = rec.n_atoms

        def cat(a, b):
            return np.concatenate([a, b])

        bonds = (np.vstack([rec.bonds, ligand.bonds + n]).astype(np.int32)
                 if len(ligand.bonds) else rec.bonds.copy())
        ms = MolSystem(
            np.vstack([rec.xyz, ligand.xyz]),
            cat(rec.element, ligand.element), bonds,
            name=cat(rec.name, ligand.name),
            hyb=cat(rec.hyb, ligand.hyb),
            aromatic=cat(rec.aromatic, ligand.aromatic),
            formal_charge=cat(rec.formal_charge, ligand.formal_charge),
            part=cat(rec.part, np.full(ligand.n_atoms, PART_LIGAND, dtype=np.int8)),
            res_name=cat(rec.res_name, ligand.res_name),
            res_seq=cat(rec.res_seq, ligand.res_seq),
            chain_id=cat(rec.chain_id, ligand.chain_id),
            meta={"n_receptor": n, "ligand_meta": dict(ligand.meta),
                  "receptor_meta": {k: v for k, v in rec.meta.items()
                                    if k in ("source", "format")}},
        )
        rc = rec.meta.get("conjugated")
        lc = ligand.meta.get("conjugated")
        if rc is not None and lc is not None:
            ms.meta["conjugated"] = cat(rc, lc)
        ms.meta["has_hydrogens"] = bool((ms.element == "H").any())
        return ms

    # ---------------------------------------------------------- perception

    def _combined_features(self, ms, n_rec):
        """Receptor features verbatim + freshly perceived ligand features.

        Receptor indices are unchanged by the merge, so its perceived features
        are reused directly rather than recomputed for every pose.
        """
        lig = ms.subset(ms.is_ligand)
        lig.meta["has_hydrogens"] = bool((lig.element == "H").any())
        lig_feats = perceive_all(lig, note_missing_hydrogens=False)

        def shift(features):
            out = []
            for f in features:
                g = f.__class__(f.kind, f.moiety_id,
                                [a + n_rec for a in f.atoms],
                                root=f.root + n_rec, note=f.note)
                out.append(g)
            return out

        merged = {}
        for key in ("donors", "acceptors", "charged", "rings", "halogens"):
            merged[key] = list(self.features[key]) + shift(lig_feats[key])
        hyd = np.zeros(ms.n_atoms, dtype=bool)
        hyd[:n_rec] = self.features["hydrophobic"]
        hyd[n_rec:] = lig_feats["hydrophobic"]
        merged["hydrophobic"] = hyd
        merged["has_hydrogens"] = ms.meta["has_hydrogens"]
        merged["warnings"] = list(self.warnings) + list(
            lig_feats.get("warnings", []))
        return merged

    # ------------------------------------------------------------- analyse

    def analyze(self, ligand: MolSystem, options: Options = None, name=None):
        opts = options or self.options
        ms = self.combine(ligand)
        n_rec = self.receptor.n_atoms
        feats = self._combined_features(ms, n_rec)

        res = detect_interactions(
            ms, enabled=opts.enabled_nci,
            include_weak_hbonds=opts.include_weak_hbonds,
            features=feats)

        if name is None:
            name = (ligand.meta.get("dock", {}).get("Name")
                    or ligand.meta.get("mol_name") or "pose")
        scores = {k: v for k, v in ligand.meta.get("dock", {}).items()
                  if isinstance(v, float)}
        scores.update({k: v for k, v in ligand.meta.get("props", {}).items()
                       if isinstance(v, float)})

        strain = None
        if opts.detect_strain:
            from .strain import analyze_strain
            strain = analyze_strain(ms)

        return PoseResult(str(name).strip(), res.interactions, res.warnings,
                          res.counts, scores, combined=ms, n_receptor=n_rec,
                          strain=strain)

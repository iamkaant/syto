"""The handbook: contextual reference data for whatever the user clicked.

The contract with the front ends is small on purpose:

    hb = Handbook.load()
    hit = hb.for_atom(molsys, atom_index)   # -> HandbookHit
    hit.highlight                           # atoms the viewer should light up
    hit.entries                             # ordered, most specific first
    hb.label_for_atom(molsys, atom_index)   # one short string for a 3D label

Both the ChimeraX panel and the web page call exactly these, so they cannot
drift apart.
"""

from __future__ import annotations

import json
import os

from ..perceive.features import classify_acceptor, classify_donor, Feature
from ..perceive import features as _F

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "handbook.json")

# Quantities worth putting directly on an atom in the 3D view, best first.
LABEL_QUANTITIES = ("pKBHX", "pKBH+", "VS_max")


class HandbookHit:
    """What the handbook has to say about one clicked atom or interaction."""

    __slots__ = ("atom", "moiety_ids", "entries", "highlight", "annotations",
                 "warnings", "title")

    def __init__(self, atom=None, moiety_ids=(), entries=(), highlight=(),
                 annotations=None, warnings=(), title=""):
        self.atom = atom
        self.moiety_ids = tuple(moiety_ids)
        self.entries = list(entries)
        self.highlight = tuple(highlight)
        self.annotations = list(annotations or [])
        self.warnings = list(warnings)
        self.title = title

    def __bool__(self):
        return bool(self.entries)

    def to_dict(self):
        return {"atom": self.atom, "title": self.title,
                "moiety_ids": list(self.moiety_ids),
                "highlight": list(self.highlight),
                "annotations": self.annotations,
                "warnings": self.warnings,
                "entries": self.entries}


class Handbook:
    def __init__(self, payload):
        self.payload = payload
        self.entries = payload.get("entries", {})
        self.by_moiety = payload.get("by_moiety", {})
        self.interaction_map = payload.get("interaction_map", {})
        self._always = payload.get("always_relevant", [])
        self.stats = payload.get("stats", {})
        self.content_hash = payload.get("content_hash", "")

    # ------------------------------------------------------------- loading

    @classmethod
    def load(cls, path=None):
        p = os.path.abspath(path or DATA_PATH)
        if not os.path.exists(p):
            raise FileNotFoundError(
                "handbook.json not found at %s.\n"
                "Build it with:  python tools/build_data.py" % p)
        with open(p, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def load_or_none(cls, path=None):
        try:
            return cls.load(path)
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------ lookups

    def entry(self, eid):
        return self.entries.get(eid)

    def always_relevant(self):
        return [self.entries[e] for e in self._always if e in self.entries]

    def for_moiety(self, moiety_id):
        """Entries for one moiety id: our narrative first, then the data."""
        rec = self.by_moiety.get(moiety_id) or {}
        out = []
        for eid in rec.get("local", []):
            if eid in self.entries:
                out.append(self.entries[eid])
        for eid in rec.get("data", []):
            if eid in self.entries:
                out.append(self.entries[eid])
        return out

    def for_atom(self, ms, idx):
        """Everything relevant to the atom the user clicked."""
        idx = int(idx)
        mids, roles = [], []

        acc = classify_acceptor(ms, idx)
        if acc:
            mids.append(acc[0])
            roles.append("acceptor: " + acc[1] if acc[1] else "acceptor")
        don = classify_donor(ms, idx)
        if don:
            mids.append(don[0])
            roles.append("donor: " + don[1] if don[1] else "donor")

        for f in _F.find_charged_groups(ms):
            if idx in f.atoms:
                mids.append(f.moiety_id)
                roles.append("%s (%s)" % (f.kind, f.note) if f.note else f.kind)
        for f in _F.find_aromatic_rings(ms):
            if idx in f.atoms:
                mids.append(f.moiety_id)
                roles.append("aromatic ring")
        for f in _F.find_halogen_donors(ms):
            if idx in f.atoms:
                mids.append(f.moiety_id)
                roles.append("halogen-bond donor")

        if not mids:
            el = str(ms.element[idx])
            return HandbookHit(
                atom=idx, title="%s - %s" % (ms.atom_label(idx), el),
                highlight=(idx,),
                warnings=["No handbook entry: this atom is not perceived as a "
                          "donor, acceptor, ring or charged group."])

        seen, entries = set(), []
        for mid in mids:
            for e in self.for_moiety(mid):
                if e["id"] in seen:
                    continue
                seen.add(e["id"])
                entries.append(e)

        highlight = self.moiety_atoms(ms, idx, mids)
        ann = self.annotations_for(mids)
        warn = []
        for e in entries:
            if e.get("curation_status") in ("needs_review",):
                warn.append("Some values here are flagged for review.")
                break

        return HandbookHit(
            atom=idx,
            title="%s - %s" % (ms.atom_label(idx), "; ".join(roles)),
            moiety_ids=mids, entries=entries, highlight=highlight,
            annotations=ann, warnings=warn)

    def for_interaction(self, interaction):
        """Entries explaining a drawn interaction line."""
        eids = list(self.interaction_map.get(interaction.kind, []))
        for mid in getattr(interaction, "moiety_ids", ()) or ():
            rec = self.by_moiety.get(mid) or {}
            eids += rec.get("local", []) + rec.get("data", [])
        seen, entries = set(), []
        for eid in eids:
            if eid in seen or eid not in self.entries:
                continue
            seen.add(eid)
            entries.append(self.entries[eid])
        return HandbookHit(
            title="%s - %s" % (interaction.criterion.label, interaction.label),
            entries=entries,
            highlight=tuple(interaction.lig_atoms) + tuple(interaction.rec_atoms),
            moiety_ids=getattr(interaction, "moiety_ids", ()))

    # -------------------------------------------------------- annotations

    def annotations_for(self, moiety_ids):
        """Numeric summaries suitable for display next to an atom."""
        out = []
        seen = set()
        for mid in moiety_ids:
            for e in self.for_moiety(mid):
                st = e.get("summary_stats") or {}
                for q in LABEL_QUANTITIES:
                    if q not in st:
                        continue
                    key = (e["id"], q)
                    if key in seen:
                        continue
                    seen.add(key)
                    s = st[q]
                    out.append({
                        "quantity": q, "entry": e["id"],
                        "entry_title": e.get("title", e["id"]),
                        "median": s["median"], "min": s["min"],
                        "max": s["max"], "n": s["n"], "unit": s.get("unit"),
                        "n_flagged": s.get("n_flagged", 0),
                        "derived": True,
                    })
        return out

    def label_for_atom(self, ms, idx, quantity="pKBHX"):
        """One compact string for a 3D atom label, or None.

        Deliberately shows the family median and range rather than a single
        number: the perception layer knows "this is a pyridine nitrogen", not
        which substituted pyridine, and those differ by whole log units.
        """
        hit = self.for_atom(ms, idx)
        for a in hit.annotations:
            if a["quantity"] != quantity:
                continue
            flag = "*" if a["n_flagged"] else ""
            if a["min"] == a["max"]:
                return "%s %.2f%s" % (quantity, a["median"], flag)
            return "%s ~%.1f (%.1f-%.1f)%s" % (
                quantity, a["median"], a["min"], a["max"], flag)
        return None

    def annotate_system(self, ms, quantity="pKBHX", ligand_only=True):
        """{atom_index: label} for every atom that has a value worth showing."""
        out = {}
        for i in range(ms.n_atoms):
            if ligand_only and not ms.is_ligand[i]:
                continue
            if str(ms.element[i]) == "H":
                continue
            lab = self.label_for_atom(ms, i, quantity)
            if lab:
                out[i] = lab
        return out

    # ----------------------------------------------------------- moieties

    def moiety_atoms(self, ms, idx, moiety_ids):
        """Which atoms to highlight for the moiety the user clicked.

        Highlighting only the clicked atom is unhelpful: the chemistry lives
        in the group. A carbonyl highlights C=O, an amide adds the nitrogen,
        a ring highlights the whole ring.
        """
        atoms = {idx}
        el = str(ms.element[idx])
        mset = set(moiety_ids)

        for f in _F.find_aromatic_rings(ms):
            if idx in f.atoms:
                atoms.update(f.atoms)
        for f in _F.find_charged_groups(ms):
            if idx in f.atoms:
                atoms.update(f.atoms)

        heavy = [int(x) for x in ms.heavy_neighbors(idx)]

        if mset & {"carbonyl_o", "amide_o", "thiocarbonyl_s"}:
            for c in heavy:
                atoms.add(c)
                for k in ms.heavy_neighbors(c):
                    if str(ms.element[int(k)]) == "N":
                        atoms.add(int(k))     # the amide nitrogen
        if mset & {"ether_o", "ester_o", "furan_o", "thioether_s"}:
            atoms.update(heavy)
        if mset & {"nitrile_n"}:
            for c in heavy:
                atoms.add(c)
                atoms.update(int(k) for k in ms.heavy_neighbors(c))
        if mset & {"sulfone_o", "sulfoxide_o", "phosphine_oxide_o",
                   "nitro_o", "n_oxide_o", "phosphate_sulfonate_o"}:
            for s in heavy:
                atoms.add(s)
                atoms.update(int(k) for k in ms.heavy_neighbors(s)
                             if str(ms.element[int(k)]) == "O")
        if mset & {"halogen_cl", "halogen_br", "halogen_i", "organic_f"}:
            atoms.update(heavy)
        # a donor is only meaningful with its hydrogen
        atoms.update(int(h) for h in ms.attached_hydrogens(idx))
        if el in ("N", "O", "S"):
            for h in heavy:
                if str(ms.element[h]) == "H":
                    atoms.add(h)
        return tuple(sorted(atoms))

    # --------------------------------------------------------------- misc

    def conflicts(self):
        """Values the extraction flagged as disagreeing between sources."""
        return self.payload.get("conflicts", [])

    def describe(self):
        s = self.stats
        return ("handbook %s: %d entries, %d values, %d flagged, %d conflicts"
                % (self.content_hash, s.get("n_entries", 0),
                   s.get("n_values", 0), s.get("n_flagged_values", 0),
                   s.get("n_conflicts", 0)))

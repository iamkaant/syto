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

from ..perceive.features import (classify_acceptor, classify_donor,
                                 classify_non_acceptor, Feature)
from ..perceive import features as _F

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "handbook.json")

# Quantities worth putting directly on an atom in the 3D view, best first.
LABEL_QUANTITIES = ("pKBHX", "pKBH+", "VS_max")

# Below this many compounds a median is not reported: the range is shown with
# its n instead. A "median" of two values 2.8 and 5.6 is just their midpoint
# and reads far more authoritative than it is.
MIN_N_FOR_MEDIAN = 5


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
        # moiety -> quantity -> the parent compound's own cited value
        self.representative = payload.get("representative", {})
        # moiety -> narrowed statistics, where a source entry lumps families
        self.subfamily = payload.get("subfamily", {})
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
        if not acc:
            neg = classify_non_acceptor(ms, idx)
            if neg:
                mids.append(neg[0])
                roles.append("NOT an acceptor - " + neg[1])

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

    def representative_for(self, moiety_ids, quantity):
        """The parent compound's own value for this moiety, if the data has it.

        Anchoring on the unsubstituted parent beats a family median: the azine
        family spans 0.3 to 2.0, while pyrazine itself is 0.92.
        """
        for mid in moiety_ids:
            rec = (self.representative.get(mid) or {}).get(quantity)
            if rec:
                return rec
        return None

    def annotations_for(self, moiety_ids):
        """Numeric summaries suitable for display next to an atom."""
        out = []
        seen = set()
        for mid in moiety_ids:
            sub = self.subfamily.get(mid) or {}
            for e in self.for_moiety(mid):
                st = e.get("summary_stats") or {}
                # Where this entry lumps several families together, use the
                # statistics for THIS moiety's family only. Quoting the whole
                # "alcohol / water / phenol" range for a phenol reads an
                # order of magnitude too basic.
                narrowed = (sub.get("stats") or {}) if sub.get("entry") == e["id"] else {}
                if narrowed:
                    st = dict(st)
                    st.update(narrowed)
                for q in LABEL_QUANTITIES:
                    if q not in st:
                        continue
                    key = (e["id"], q)
                    if key in seen:
                        continue
                    seen.add(key)
                    s = st[q]
                    ann = {
                        "quantity": q, "entry": e["id"],
                        "entry_title": e.get("title", e["id"]),
                        "median": s["median"], "min": s["min"],
                        "max": s["max"], "n": s["n"], "unit": s.get("unit"),
                        "n_flagged": s.get("n_flagged", 0),
                        "derived": True,
                        "narrowed": bool(narrowed.get(q)),
                    }
                    rep = self.representative_for(moiety_ids, q)
                    if rep and rep.get("entry") == e["id"]:
                        ann["representative"] = rep
                    # What the quantity measures, and - for a proton
                    # basicity with a value behind it - what that means for
                    # the pose. The card has to stand on its own: the scale
                    # plot carries the same verdict, and the plots can be
                    # switched off.
                    from .scale import quantity_info, protonation_verdict
                    ann["means"] = quantity_info(q).get("means") or ""
                    # `changes_the_answer` is what a front end shows by
                    # default. For most quantities the number IS the answer.
                    #
                    # pKBH+ is different: it is a precondition, not an
                    # opinion. Where the atom is neutral at pH 7.4 it adds
                    # nothing that pKBHX did not already say, and a second
                    # plot on every click is noise. Where it is protonated,
                    # the pKBHX value is for a species that is not in the
                    # pocket at all - the atom is a cation that donates.
                    #
                    # Measured over the real ligands in three saved
                    # sessions: 0 of 240 checkable atoms in one, 5 of 53 in
                    # another, and 55 of 359 in a third, where the common
                    # offenders are amidines and tertiary amines - exactly
                    # the groups a picker stops to look at.
                    ann["changes_the_answer"] = True
                    if q == "pKBH+":
                        v = (rep or {}).get("value")
                        if v is None and s["n"] >= MIN_N_FOR_MEDIAN:
                            v = s["median"]
                        if isinstance(v, (int, float)):
                            ann["verdict"] = protonation_verdict(v)
                        ann["changes_the_answer"] = bool(
                            ann.get("verdict")
                            and ann["verdict"]["level"] != "neutral")
                    out.append(ann)
        return out

    def label_for_atom(self, ms, idx, quantity="pKBHX"):
        """One compact string for a 3D atom label, or None.

        Prefers the parent compound's own cited value and keeps the family
        range beside it, so the number is anchored on something real while
        still showing how far substitution can move it.
        """
        hit = self.for_atom(ms, idx)
        for a in hit.annotations:
            if a["quantity"] != quantity:
                continue
            flag = "*" if a["n_flagged"] else ""
            rep = a.get("representative")
            if rep and isinstance(rep.get("value"), (int, float)):
                rflag = "*" if rep.get("flags") else ""
                head = "%s %.2f%s" % (quantity, float(rep["value"]), rflag)
                # Only show a range when there is a real range to show: with
                # one compound behind it, "(-0.4--0.4)" is noise, and the
                # double minus is unreadable.
                if a["n"] < 2 or round(a["min"], 1) == round(a["max"], 1):
                    return head
                return "%s (%.1f to %.1f)%s" % (head, a["min"], a["max"], flag)
            n = a["n"]
            if a["min"] == a["max"]:
                return "%s %.2f%s" % (quantity, a["median"], flag)
            if n < MIN_N_FOR_MEDIAN:
                # Too few compounds for a median to mean anything.
                return "%s %.1f to %.1f (n=%d)%s" % (
                    quantity, a["min"], a["max"], n, flag)
            return "%s ~%.1f (%.1f to %.1f)%s" % (
                quantity, a["median"], a["min"], a["max"], flag)
        return None

    def annotate_system(self, ms, quantity="pKBHX", ligand_only=True,
                        heteroatoms_only=True, deduplicate=True):
        """{atom_index: label} for atoms worth annotating in 3D.

        Two filters, both on by default, because labelling everything makes
        the view unreadable and buries the values that matter:

        * `heteroatoms_only` skips carbon. An aromatic carbon does carry a
          pKBHX (benzene is a very weak pi acceptor, about -0.5), but putting
          that on all six carbons of every ring drowns out the pyridine
          nitrogen next to it.
        * `deduplicate` emits one label per moiety rather than one per atom,
          so a ring gets a single annotation instead of six identical ones.
        """
        out = {}
        seen = set()
        for i in range(ms.n_atoms):
            if ligand_only and not ms.is_ligand[i]:
                continue
            el = str(ms.element[i])
            if el == "H":
                continue
            if heteroatoms_only and el == "C":
                continue
            lab = self.label_for_atom(ms, i, quantity)
            if not lab:
                continue
            if deduplicate:
                hit = self.for_atom(ms, i)
                key = (lab, frozenset(hit.highlight))
                if key in seen:
                    continue
                seen.add(key)
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

    def scale(self, quantity="pKBHX", moiety_ids=(), same_element_only=False):
        """Where a moiety sits among every family on the `quantity` scale."""
        from .scale import build_scale
        return build_scale(self, quantity, moiety_ids, same_element_only)

    def scale_text(self, quantity="pKBHX", moiety_ids=(),
                   same_element_only=True, width=46):
        from .scale import build_scale, text_scale
        return text_scale(build_scale(self, quantity, moiety_ids,
                                      same_element_only), width)

    def conflicts(self):
        """Values the extraction flagged as disagreeing between sources."""
        return self.payload.get("conflicts", [])

    def describe(self):
        s = self.stats
        return ("handbook %s: %d entries, %d values, %d flagged, %d conflicts"
                % (self.content_hash, s.get("n_entries", 0),
                   s.get("n_values", 0), s.get("n_flagged_values", 0),
                   s.get("n_conflicts", 0)))

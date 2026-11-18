#!/usr/bin/env python3
"""Compile data-src/ into the versioned handbook.json the front ends read.

Two kinds of input are merged:

  data-src/*.yaml              our own narrative entries: what a moiety is,
                               why it matters, the negative cases, practical
                               advice. Few numbers.
  data-src/external/*.yaml     extracted literature parameters, every value
                               carrying a DOI and a locator.

They are joined through data-src/moiety_map.yaml, which maps what the
perception layer emits onto the source dataset's finer vocabulary.

The build is the quality gate. It FAILS when:
  * a numeric value has no DOI, or a DOI but no locator
  * an entry names a moiety_id perception never produces
  * the moiety map points at an entry id that does not exist
  * --release is passed and anything is still unverified

Usage:
    python tools/build_data.py            # build, warn about what needs review
    python tools/build_data.py --check    # validate only, write nothing
    python tools/build_data.py --release  # refuse to ship uncurated data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data-src")
EXT = os.path.join(SRC, "external")
OUT = os.path.join(ROOT, "core", "hitpicking", "data", "handbook.json")

sys.path.insert(0, os.path.join(ROOT, "core"))

VALID_STATUS = {"verified", "needs_review", "pending", "extracted"}
# Flags that mean "do not present this as a settled measurement".
CAUTION_FLAGS = {"crystal_statistics", "gas_phase_only", "value_conflict",
                 "secondary_source", "secondary_citation", "read_from_image",
                 "computed"}


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required.\n  conda install pyyaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def known_moiety_ids():
    """Every moiety id perception can emit, from the registry in features.py."""
    from hitpicking.perceive.features import MOIETY_IDS
    return set(MOIETY_IDS)


def iter_values(entry):
    return entry.get("values") or []


def validate_entry(e, known, errors, warnings, require_moiety=True):
    eid = e.get("id")
    if not eid:
        errors.append("an entry has no id")
        return
    if not e.get("title"):
        warnings.append("%s: no title" % eid)

    status = e.get("curation_status", "needs_review")
    if status not in VALID_STATUS:
        errors.append("%s: bad curation_status %r" % (eid, status))

    if require_moiety:
        mids = e.get("moiety_ids") or []
        if not mids:
            warnings.append("%s: no moiety_ids; it can never be shown "
                            "contextually" % eid)
        for mid in mids:
            if mid not in known:
                errors.append("%s: moiety_id %r is never produced by "
                              "perception" % (eid, mid))

    for v in iter_values(e):
        q = v.get("quantity", "?")
        vstatus = v.get("curation_status", status)
        ref = v.get("ref") or {}
        has_value = v.get("value") is not None
        if has_value:
            if not ref.get("doi"):
                errors.append("%s/%s: a value with no DOI" % (eid, q))
            elif not ref.get("locator"):
                errors.append("%s/%s: DOI but no locator" % (eid, q))
            if vstatus == "pending":
                errors.append("%s/%s: has a value but is marked pending"
                              % (eid, q))
        elif vstatus != "pending":
            warnings.append("%s/%s: no value but not marked pending" % (eid, q))

    geom = e.get("geometry") or {}
    d = [geom.get(k) for k in ("d_min", "d_ideal", "d_max")]
    if all(x is not None for x in d) and not (d[0] < d[1] <= d[2]):
        errors.append("%s: need d_min < d_ideal <= d_max, got %s" % (eid, d))


def summarise_quantities(entry):
    """Per-entry distribution of each numeric quantity.

    Derived from the cited values, not a new measurement: it is what lets a
    3D label read "pKBHX 1.86 (pyridines: 0.4-3.0, n=48)" instead of dumping
    48 compounds onto an atom. Values carrying a caution flag are counted
    separately so the UI can say how much of the family is still unreviewed.
    """
    out = {}
    for v in iter_values(entry):
        q = v.get("quantity")
        val = v.get("value")
        if q is None or val is None:
            continue
        nums = val if isinstance(val, list) else [val]
        nums = [float(x) for x in nums
                if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if not nums:
            continue
        d = out.setdefault(q, {"values": [], "unit": v.get("unit"),
                               "n_flagged": 0, "examples": []})
        d["values"].extend(nums)
        if set(v.get("flags") or []) & CAUTION_FLAGS:
            d["n_flagged"] += 1
        if v.get("compound") and len(d["examples"]) < 6:
            d["examples"].append({"compound": v["compound"],
                                  "value": nums[0] if len(nums) == 1 else nums})
    summary = {}
    for q, d in out.items():
        vals = sorted(d["values"])
        n = len(vals)
        mid = n // 2
        median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0
        summary[q] = {
            "n": n, "min": round(vals[0], 3), "max": round(vals[-1], 3),
            "median": round(median, 3), "unit": d["unit"],
            "n_flagged": d["n_flagged"], "examples": d["examples"],
            "derived": True,
        }
    return summary


def build(strict_release=False, write=True, quiet=False):
    if not os.path.isdir(SRC):
        sys.exit("no data-src/ at %s" % SRC)

    known = known_moiety_ids()
    errors, warnings = [], []

    # ---- our narrative entries -------------------------------------------
    local, sources = [], []
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith((".yaml", ".yml")) or fn == "moiety_map.yaml":
            continue
        doc = load_yaml(os.path.join(SRC, fn)) or {}
        sources.append(fn)
        for e in doc.get("entries", []) or []:
            e.setdefault("source_file", fn)
            e.setdefault("origin", "local")
            local.append(e)
            validate_entry(e, known, errors, warnings)

    # ---- extracted literature parameters ---------------------------------
    external, ext_meta = [], {}
    if os.path.isdir(EXT):
        for fn in sorted(os.listdir(EXT)):
            if not fn.endswith((".yaml", ".yml")):
                continue
            doc = load_yaml(os.path.join(EXT, fn)) or {}
            sources.append("external/" + fn)
            ext_meta[fn] = doc.get("schema_notes", {})
            for e in doc.get("entries", []) or []:
                e.setdefault("source_file", "external/" + fn)
                e.setdefault("origin", "extracted")
                external.append(e)
                # the external vocabulary is intentionally finer than
                # perception's, so its moiety_ids are not checked against it
                validate_entry(e, known, errors, warnings,
                               require_moiety=False)

    ext_by_id = {e["id"]: e for e in external}

    # ---- the join --------------------------------------------------------
    mp = load_yaml(os.path.join(SRC, "moiety_map.yaml")) or {}
    moiety_map = mp.get("map", {}) or {}
    always = mp.get("always_relevant", []) or []
    inter_map = mp.get("interaction_map", {}) or {}

    for mid, eids in moiety_map.items():
        if mid not in known:
            errors.append("moiety_map: %r is not a perception moiety id" % mid)
        for eid in eids:
            if eid not in ext_by_id:
                errors.append("moiety_map[%s]: no such entry %r" % (mid, eid))
    for eid in always:
        if eid not in ext_by_id:
            errors.append("always_relevant: no such entry %r" % eid)
    for kind, eids in inter_map.items():
        for eid in eids:
            if eid not in ext_by_id:
                errors.append("interaction_map[%s]: no such entry %r"
                              % (kind, eid))

    # local entries contribute their own moiety ids too
    by_moiety = {}
    for e in local:
        for mid in e.get("moiety_ids") or []:
            by_moiety.setdefault(mid, {"local": [], "data": []})["local"].append(e["id"])
    for mid, eids in moiety_map.items():
        by_moiety.setdefault(mid, {"local": [], "data": []})["data"].extend(eids)

    if strict_release:
        for e in local + external:
            if e.get("curation_status") not in ("verified",):
                errors.append("%s: not verified (release build)" % e["id"])

    if not quiet:
        for w in warnings[:40]:
            print("  warning: %s" % w)
        if len(warnings) > 40:
            print("  ... and %d more warnings" % (len(warnings) - 40))
    if errors:
        print("\n%d error(s):" % len(errors), file=sys.stderr)
        for e in errors[:40]:
            print("  ERROR: %s" % e, file=sys.stderr)
        sys.exit(1)

    # ---- stats -----------------------------------------------------------
    n_values = sum(len(iter_values(e)) for e in local + external)
    n_flagged = 0
    flag_counts = {}
    conflicts = []
    for e in external:
        for v in iter_values(e):
            fl = set(v.get("flags") or [])
            if fl & CAUTION_FLAGS:
                n_flagged += 1
            for f in fl:
                flag_counts[f] = flag_counts.get(f, 0) + 1
            if "value_conflict" in fl:
                conflicts.append({
                    "entry": e["id"], "quantity": v.get("quantity"),
                    "compound": v.get("compound"), "value": v.get("value"),
                    "ref": v.get("ref"), "note": v.get("note"),
                })

    # resolve each moiety's parent compound to a concrete cited value
    representative = {}
    for mid, compound in (mp.get("representative", {}) or {}).items():
        if mid not in moiety_map:
            errors.append("representative[%s]: not in the moiety map" % mid)
            continue
        target = str(compound).strip().lower()
        hits = []
        for eid in moiety_map[mid]:
            for v in iter_values(ext_by_id.get(eid, {})):
                if str(v.get("compound", "")).strip().lower() != target:
                    continue
                if v.get("value") is None:
                    continue
                hits.append({"entry": eid, "quantity": v.get("quantity"),
                             "value": v.get("value"), "unit": v.get("unit"),
                             "compound": v.get("compound"),
                             "ref": v.get("ref"), "flags": v.get("flags") or []})
        if not hits:
            warnings.append("representative[%s]: no value named %r in %s"
                            % (mid, compound, moiety_map[mid]))
            continue
        by_q = {}
        for h in hits:
            by_q.setdefault(h["quantity"], []).append(h)
        resolved = {}
        for q, lst in by_q.items():
            vals = {str(x["value"]) for x in lst}
            rec = dict(lst[0])
            if len(vals) > 1:
                # the sources disagree about this very compound; say so
                rec["disagrees"] = sorted(vals)
                rec["flags"] = list(set(rec["flags"]) | {"value_conflict"})
            resolved[q] = rec
        representative[mid] = resolved

    entries = {e["id"]: e for e in local + external}
    for e in entries.values():
        st = summarise_quantities(e)
        if st:
            e["summary_stats"] = st

    payload = {
        "schema_version": 1,
        "entries": entries,
        "by_moiety": by_moiety,
        "representative": representative,
        "always_relevant": always,
        "interaction_map": inter_map,
        "sources": sources,
        "external_schema_notes": ext_meta,
        "conflicts": conflicts,
        "stats": {
            "n_entries": len(entries),
            "n_local": len(local),
            "n_extracted": len(external),
            "n_values": n_values,
            "n_flagged_values": n_flagged,
            "n_conflicts": len(conflicts),
            "flags": flag_counts,
            "n_representative": len(representative),
            "n_mapped_moieties": len(moiety_map),
            "n_unmapped_moieties": len(known - set(moiety_map)),
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["content_hash"] = hashlib.sha256(blob.encode()).hexdigest()[:16]

    if not quiet:
        s = payload["stats"]
        print("\n%d entries (%d narrative + %d extracted), %d values"
              % (s["n_entries"], s["n_local"], s["n_extracted"], s["n_values"]))
        print("%d values carry a caution flag; %d are value conflicts"
              % (s["n_flagged_values"], s["n_conflicts"]))
        print("flags: %s" % s["flags"])
        print("%d/%d perception moieties mapped to literature data"
              % (s["n_mapped_moieties"],
                 s["n_mapped_moieties"] + s["n_unmapped_moieties"]))
        unmapped = sorted(known - set(moiety_map))
        if unmapped:
            print("unmapped: %s" % ", ".join(unmapped))

    if write:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        size = os.path.getsize(OUT) / 1024.0
        print("wrote %s  (%s, %.0f KB)"
              % (os.path.relpath(OUT, ROOT), payload["content_hash"], size))
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only")
    ap.add_argument("--release", action="store_true",
                    help="fail if anything is still unverified")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    build(strict_release=a.release, write=not a.check, quiet=a.quiet)


if __name__ == "__main__":
    main()

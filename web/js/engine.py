"""The web page's view of the shared core, run inside Pyodide.

Every number the page shows comes from here, and `hitpicking` is the same
package the ChimeraX bundle uses - not a port and not a reimplementation,
so there is exactly one place where an interaction is decided. Anything
this file starts deciding for itself is a place the two front ends can
drift, which is why it does as little as possible: load files, call
ReceptorContext.analyze, and hand back plain JSON.

Files arrive as text and are written to Pyodide's virtual filesystem,
because the core's readers take paths. That is deliberate: it keeps the
browser from needing its own parser.
"""
import json
import math
import os

from hitpicking.analysis import ReceptorContext, Options
from hitpicking.handbook import Handbook
from hitpicking.io.pdb import read_pdb
from hitpicking.io.mol2 import read_mol2, read_mol2_iter
from hitpicking.io.sdf import read_sdf, read_sdf_iter

WORK = "/work"
_STATE = {"context": None, "handbook": None, "poses": [], "receptor": None}


def _finite(obj):
    """Replace NaN and infinity with null, recursively.

    An undefined angle is NaN in the core, and Python writes that as a bare
    `NaN` token, which is not JSON and which JSON.parse refuses outright.
    The whole result then arrives in the browser as an unparsed string that
    looks like a bridge bug. Null is the honest crossing for "there is no
    value here", and it is what the panels already test for.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _json(obj):
    # allow_nan=False so a non-finite value that slips past _finite raises
    # here, where the traceback names the field, instead of becoming an
    # unparseable string in the page.
    return json.dumps(_finite(obj), allow_nan=False)


def _path(name, text):
    os.makedirs(WORK, exist_ok=True)
    p = os.path.join(WORK, os.path.basename(name))
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


def handbook():
    if _STATE["handbook"] is None:
        _STATE["handbook"] = Handbook.load_or_none()
    return _STATE["handbook"]


def _read_ligands(path, limit=None):
    low = path.lower()
    if low.endswith(".mol2"):
        it = read_mol2_iter(path)
    elif low.endswith((".sdf", ".sd", ".mol")):
        it = read_sdf_iter(path)
    else:
        raise ValueError("ligands must be .mol2 or .sdf, not %s" % path)
    out = []
    for ms in it:
        out.append(ms)
        if limit and len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ set up

def setup(receptor_name, receptor_text):
    """Perceive the receptor once; every pose is then cheap."""
    p = _path(receptor_name, receptor_text)
    rec = read_pdb(p)
    ctx = ReceptorContext(rec, Options())
    _STATE["context"] = ctx
    _STATE["receptor"] = receptor_name
    hb = handbook()
    return _json({
        "receptor": receptor_name,
        "n_atoms": int(rec.n_atoms),
        "has_hydrogens": bool(ctx.has_hydrogens),
        "warnings": list(ctx.warnings),
        "handbook": hb.describe() if hb else None,
    })


def load_poses(name, text, limit=None):
    if _STATE["context"] is None:
        raise RuntimeError("load a receptor first")
    p = _path(name, text)
    poses = _read_ligands(p, int(limit) if limit else None)
    _STATE["poses"] = poses
    return _json({
        "file": name,
        "count": len(poses),
        "names": [_pose_name(m, i) for i, m in enumerate(poses)],
    })


def _pose_name(ms, i):
    d = ms.meta.get("dock") or {}
    return str(d.get("Name") or ms.meta.get("mol_name") or "pose %d" % (i + 1))


# ---------------------------------------------------------------- analyse

def _maybe_list(value):
    """JavaScript null arrives as JsNull, which is not None and not iterable.

    Everything crossing from the page goes through here rather than being
    trusted to be a Python type, because the failure mode is a TypeError
    deep inside the core rather than a clear complaint at the boundary.
    """
    if value is None:
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    return items or None


def analyse(index=0, enabled=None, strain=True):
    ctx = _STATE["context"]
    if ctx is None:
        raise RuntimeError("load a receptor first")
    poses = _STATE["poses"]
    if not poses:
        raise RuntimeError("load poses first")
    i = int(index)
    if not (0 <= i < len(poses)):
        raise IndexError("pose %d of %d" % (i, len(poses)))
    opts = Options()
    kinds = _maybe_list(enabled)
    if kinds:
        opts.enabled_nci = set(kinds)
    opts.detect_strain = bool(strain)
    res = ctx.analyze(poses[i], options=opts, name=_pose_name(poses[i], i))
    _STATE["last"] = res
    return _json(_pose_dict(res, i))


def _pose_dict(res, index):
    out = {
        "index": index,
        "name": res.name,
        "counts": dict(res.counts),
        "warnings": list(res.warnings),
        "interactions": [i.to_dict() for i in res.interactions],
        "n_receptor": int(res.n_receptor),
        "summary": res.summary(),
    }
    sr = res.strain
    if sr is not None:
        out["strain"] = {
            "counts": dict(sr.counts),
            "worst_overlap": round(float(sr.worst_overlap), 3),
            "clashes": [{"kind": c.kind, "distance": round(c.distance, 3),
                         "overlap": round(c.overlap, 3),
                         "severe": bool(c.severe), "label": c.label}
                        for c in (sr.clashes + sr.internal)],
            "findings": [{"kind": f.kind, "value": f.value, "unit": f.unit,
                          "message": f.message, "severity": f.severity,
                          "atoms": [int(a) for a in f.atoms]}
                         for f in sr.findings],
        }
    return out


# ------------------------------------------------------------------ click

def explain(index, atom):
    """The handbook entry for one atom of the analysed pose."""
    hb = handbook()
    if hb is None:
        return _json({"error": "handbook data not built"})
    res = _STATE.get("last")
    if res is None or res.combined is None:
        raise RuntimeError("analyse a pose first")
    hit = hb.for_atom(res.combined, int(atom))
    d = hit.to_dict()
    d["scales"] = _scales(hb, hit)
    return _json(d)


def _scales(hb, hit):
    """pKBHX always; pKBH+ only where it overrides it - as in ChimeraX."""
    wanted = {"pKBHX"}
    for a in hit.annotations:
        if a.get("changes_the_answer"):
            wanted.add(a["quantity"])
    out = []
    for q in ("pKBHX", "pKBH+"):
        if q not in wanted:
            continue
        sc = hb.scale(q, hit.moiety_ids, same_element_only=False)
        if sc:
            out.append(sc)
    return out


def version():
    import hitpicking
    hb = handbook()
    return _json({
        "feature_version": getattr(hitpicking, "FEATURE_VERSION", None),
        "core_version": getattr(hitpicking, "__version__", None),
        "handbook": hb.describe() if hb else None,
        "handbook_hash": hb.content_hash if hb else None,
    })

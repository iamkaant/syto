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
        # Which interaction types the core draws unless told otherwise.
        # The page used to assume "all of them" and tick every box,
        # which was a lie about what was being computed - and the first
        # toggle sent those ticks back as the truth, switching on types
        # that had never been on.
        "default_nci": sorted(Options().enabled_nci),
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


def analyse(index=0, enabled=None, strain=True, stranded_donors=True,
            stranded_acceptors=False):
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
    opts.stranded_donors = bool(stranded_donors)
    opts.stranded_acceptors = bool(stranded_acceptors)
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
    if res.unsatisfied is not None:
        out["unsatisfied"] = [u.to_dict() for u in res.unsatisfied]
    return out


# ------------------------------------------------- active learning (wip)

def _learner():
    """The one learner for this tab.

    Created on first use so a page that only wants the handbook never
    imports the model, its features or numpy's linear algebra paths -
    which is the same reason the ChimeraX bundle defers them.
    """
    if _STATE.get("learner") is None:
        from hitpicking.al import Learner
        _STATE["learner"] = Learner(seed=0)
    return _STATE["learner"]


def featurise(start=0, count=1, strain=True):
    """Analyse poses and turn them into feature vectors.

    This is the slow step of a hit-picking party and the one that decides
    whether the web page can host one: a few thousand poses have to go
    through the full interaction and strain analysis before the model has
    anything to learn from. Reported with its own timing so the cost is a
    measurement rather than a guess.
    """
    import time
    ctx = _STATE["context"]
    if ctx is None:
        raise RuntimeError("load a receptor first")
    poses = _STATE["poses"]
    if not poses:
        raise RuntimeError("load poses first")
    from hitpicking.al.features import pose_features

    lr = _learner()
    opts = Options()
    opts.detect_strain = bool(strain)

    start = max(0, int(start))
    end = min(len(poses), start + int(count))
    t0 = time.perf_counter()
    t_analyse = 0.0
    done = 0
    for i in range(start, end):
        name = _pose_name(poses[i], i)
        a0 = time.perf_counter()
        res = ctx.analyze(poses[i], options=opts, name=name)
        t_analyse += time.perf_counter() - a0
        key = _key(i, name)
        lr.add_features(key, pose_features(res, terms=lr.score_terms),
                        res.scores)
        _STATE.setdefault("keys", {})[i] = key
        # A grade given before this pose was featurised is still a grade;
        # hand it over now rather than losing it.
        pending = (_STATE.get("grades") or {}).get(key)
        if pending:
            lr.grade(key, pending)
        _STATE.setdefault("summary", {})[i] = {
            "name": name, "summary": res.summary(),
            "counts": dict(res.counts),
            "worst_overlap": (round(float(res.strain.worst_overlap), 2)
                              if res.strain is not None else None),
        }
        done += 1
    total = time.perf_counter() - t0

    return _json({
        "featurised": done,
        "total_known": len(lr.features),
        "of": len(poses),
        "next": end,
        "seconds": round(total, 3),
        "per_pose": round(total / done, 4) if done else None,
        "analyse_share": round(t_analyse / total, 3) if total else None,
    })


def _key(i, name=None):
    """A pose's identity for the learner.

    The index alone would not survive a saved party being reloaded
    against a re-ordered file, and the name alone is not unique - a mol2
    can hold the same compound in several poses. Both, so a grade given
    today still means something tomorrow.
    """
    if name is None:
        poses = _STATE["poses"]
        name = _pose_name(poses[i], i) if 0 <= i < len(poses) else "?"
    return "%d|%s" % (int(i), name)


def _index_of(key):
    for i, k in (_STATE.get("keys") or {}).items():
        if k == key:
            return int(i)
    head = str(key).split("|", 1)[0]
    return int(head) if head.isdigit() else -1


def grade(index, value):
    """Record one human judgement. `value` is bad / maybe / good, or None.

    A grade is a person's opinion about a pose. It does not need a model,
    a feature vector, or anything else to exist - and requiring one meant
    you could not simply walk a list and write down what you thought,
    which is what a hit-picking party mostly is. So the grade is recorded
    here regardless, and handed to the learner if and when that pose is
    featurised.
    """
    i = int(index)
    poses = _STATE["poses"]
    if not (0 <= i < len(poses)):
        raise IndexError("pose %d of %d" % (i, len(poses)))
    key = _key(i)
    grades = _STATE.setdefault("grades", {})
    if value:
        grades[key] = str(value)
    else:
        grades.pop(key, None)
    lr = _STATE.get("learner")
    if lr is not None and key in lr.features:
        lr.grade(key, value if value else None)
    return al_status()


def train(epochs=100, batch_size=None, patience=20):
    import time
    lr = _learner()
    ok, why = lr.can_train()
    if not ok:
        return _json({"trained": False, "why_not": why})
    t0 = time.perf_counter()
    lr.train(epochs=int(epochs),
             batch_size=int(batch_size) if batch_size else lr.DEFAULT_BATCH,
             patience=int(patience))
    secs = time.perf_counter() - t0
    d = json.loads(al_status())
    d["trained"] = True
    d["seconds"] = round(secs, 2)
    d["trained_on"] = lr.trained_on
    d["block_variance"] = {k: round(v, 3)
                           for k, v in lr.block_variance().items()}
    return _json(d)


def predictions(strategy="uncertainty", count=25):
    """Every featurised pose, scored, plus what to look at next.

    `rows` is the whole set so the pose list can be sorted by it -  a 25
    row panel is the wrong shape for picking from a few thousand poses,
    which is the lesson the ChimeraX side already learned.
    """
    lr = _STATE.get("learner")
    if lr is None or lr.committee is None:
        return _json({"ready": False,
                      "why_not": "the model has not been trained yet"})
    from hitpicking.al import DISAGREEMENT, PREDICTION, UNCERTAINTY
    strat = {"uncertainty": UNCERTAINTY, "disagreement": DISAGREEMENT,
             "prediction": PREDICTION}.get(str(strategy), UNCERTAINTY)

    preds = lr.predict()
    rows = []
    for key, p in preds.items():
        i = _index_of(key)
        rows.append({"index": i, "key": key, "name": key.split("|", 1)[-1],
                     "grade": p["grade"], "score": round(p["score"], 3),
                     "uncertainty": round(p["uncertainty"], 4),
                     "error": p["error"],
                     "yours": lr.grades.label(key)})
    rows.sort(key=lambda r: -r["score"])

    ranked = lr.rank_poses(strategy=strat,
                           exclude_graded=(strat != DISAGREEMENT))
    nxt = []
    for key, p in ranked[:int(count)]:
        nxt.append({"index": _index_of(key), "name": key.split("|", 1)[-1],
                    "grade": p["grade"], "score": round(p["score"], 3),
                    "uncertainty": round(p["uncertainty"], 4),
                    "yours": lr.grades.label(key), "error": p["error"]})
    return _json({"ready": True, "strategy": strategy,
                  "rows": rows, "next": nxt})


def save_party():
    """The grades and the model, as JSON the page can hand to the user.

    Feature vectors are deliberately not in here: they are reproducible
    from the files in seconds and would make the download enormous. This
    is the same thing `hitpick learn save` writes, so a party started in
    the browser can be finished in ChimeraX.
    """
    lr = _STATE.get("learner")
    human = dict(_STATE.get("grades") or {})
    if lr is None:
        # A party with opinions and no model is a perfectly good party.
        # Refusing to save it was the model's needs standing in for the
        # user's, which is the thing this whole change is about.
        if not human:
            raise RuntimeError("nothing to save yet")
        return _json({"feature_version": None, "grades": {},
                      "human_grades": human,
                      "poses": {str(i): _key(i)
                                for i in range(len(_STATE.get("poses") or []))},
                      "receptor": _STATE.get("receptor"),
                      "committee": None})
    st = lr.state()
    st["poses"] = {str(i): k for i, k in (_STATE.get("keys") or {}).items()}
    st["receptor"] = _STATE.get("receptor")
    # Every opinion, including the ones on poses nobody has featurised.
    # Those are the whole point of grading without the model.
    st["human_grades"] = human
    return _json(st)


def load_party(text):
    """Grades and model back in, matched to the poses now loaded.

    The grades are keyed on index-and-name, so a grade is only restored
    to a pose that is still the same pose. Anything that does not match
    is reported rather than quietly dropped - a party that loses half its
    opinions in silence is worse than one that refuses to load.
    """
    from hitpicking.al import Learner
    data = json.loads(text) if isinstance(text, str) else dict(text)
    human = dict(data.get("human_grades") or {})
    if data.get("feature_version") is None and not data.get("committee"):
        # grades only, saved before anything was analysed
        lr = _STATE.get("learner")
    else:
        lr = Learner.from_state(data)
        _STATE["learner"] = lr
    if lr is not None and not human:
        # a party saved before grades were kept separately
        human = {k: lr.grades.options[v]
                 for k, v in (data.get("grades") or {}).items()
                 if 0 <= int(v) < len(lr.grades.options)}
    _STATE["grades"] = human
    if lr is not None:
        for key, value in human.items():
            if key in lr.features:
                lr.grade(key, value)

    have = {}
    poses = _STATE["poses"]
    for i in range(len(poses)):
        have[_key(i)] = i
    matched, orphan = 0, []
    for key in list(human):
        if key in have:
            matched += 1
        else:
            orphan.append(key)
    d = json.loads(al_status())
    d["restored"] = matched
    d["unmatched"] = len(orphan)
    d["unmatched_names"] = [k.split("|", 1)[-1] for k in orphan[:8]]
    return _json(d)


def al_status():
    """Where the party is. Human grades and model state are separate.

    `graded` is how many opinions have been recorded - the thing a party
    actually produces. `graded_featurised` is how many of those the model
    can learn from, which is a different and much less interesting number
    until somebody wants to train.
    """
    human = _STATE.get("grades") or {}
    counts = {"bad": 0, "maybe": 0, "good": 0}
    for v in human.values():
        if v in counts:
            counts[v] += 1
    lr = _STATE.get("learner")
    out = {"poses": 0 if lr is None else len(lr.features),
           "of": len(_STATE.get("poses") or []),
           "graded": len(human), "grades": counts,
           "graded_featurised": 0 if lr is None else lr.n_graded,
           "trained": bool(lr is not None and lr.committee is not None),
           "can_train": False, "why_not": "", "score_terms": []}
    if lr is not None:
        ok, why = lr.can_train()
        out["can_train"] = ok
        out["why_not"] = why
        out["score_terms"] = list(lr.score_terms)
    elif human:
        out["why_not"] = ("press Analyse all poses before training - the "
                          "model needs the features, though your grades "
                          "are being kept either way")
    return _json(out)


def export_rows():
    """The party's result: one row a pose, as a table.

    This is what leaves the building. The saved party is for carrying on
    later; this is the list you take to the meeting - what you said,
    what the model thinks if it was trained, and what the pose actually
    does if it was analysed.
    """
    poses = _STATE.get("poses") or []
    grades = _STATE.get("grades") or {}
    summ = _STATE.get("summary") or {}
    lr = _STATE.get("learner")
    preds = {}
    if lr is not None and lr.committee is not None:
        try:
            for k, p in lr.predict().items():
                preds[k] = p
        except Exception:
            preds = {}

    rows = []
    for i in range(len(poses)):
        key = _key(i)
        p = preds.get(key) or {}
        d = summ.get(i) or {}
        c = d.get("counts") or {}
        rows.append({
            "pose": i + 1,
            "name": _pose_name(poses[i], i),
            "your_grade": grades.get(key, ""),
            "predicted": p.get("grade", ""),
            "score": "" if p.get("score") is None else round(p["score"], 3),
            "uncertainty": ("" if p.get("uncertainty") is None
                            else round(p["uncertainty"], 4)),
            "hbonds": c.get("hbond", 0) + c.get("hbond_charged", 0),
            "salt_bridges": c.get("salt_bridge", 0),
            "stacking": (c.get("pi_stack_parallel", 0)
                         + c.get("pi_stack_tshaped", 0)),
            "worst_overlap": d.get("worst_overlap", ""),
            "summary": d.get("summary", ""),
        })
    return _json({"rows": rows,
                  "columns": ["pose", "name", "your_grade", "predicted",
                              "score", "uncertainty", "hbonds",
                              "salt_bridges", "stacking", "worst_overlap",
                              "summary"],
                  "receptor": _STATE.get("receptor")})


# ------------------------------------------------------------------ scene

# Interactions whose geometry is about a ring as a whole: the distance
# reported for these was measured to a centroid, not to an atom.
CENTROID_KINDS = frozenset(("pi_stack_parallel", "pi_stack_tshaped",
                            "cation_pi"))


def scene(index=None):
    """What to draw, as coordinates, for the last analysed pose.

    Endpoints rather than atom indices: the page's viewer parses the same
    files with its own reader and numbers the atoms its own way, and a
    shared index would be a silent assumption that two parsers agree.
    Coordinates are read verbatim from the file by both, so they are the
    one thing that cannot drift.

    Nothing here decides anything. The interactions, their quality and
    their labels are exactly what `analyse` already returned; this is the
    same list expressed as geometry.
    """
    res = _STATE.get("last")
    if res is None or res.combined is None:
        raise RuntimeError("analyse a pose first")
    ms = res.combined

    def xyz(i):
        v = ms.xyz[int(i)]
        return [round(float(v[0]), 3), round(float(v[1]), 3),
                round(float(v[2]), 3)]

    # Which side of which interaction is a ring, so the page can draw
    # the line where the measurement was actually made. A stack measured
    # between two ring centroids but drawn from one ring atom to another
    # is a picture of something that was never computed - and it looks
    # like a bad stack when the numbers say it is a good one.
    ring_sets = set()
    try:
        for r in ms.rings():
            ring_sets.add(frozenset(int(a) for a in r))
    except Exception:
        pass

    def endpoint(atoms):
        """(coordinates to span, whether they are a ring)."""
        ats = [int(a) for a in atoms]
        if len(ats) >= 5 and frozenset(ats) in ring_sets:
            return [xyz(a) for a in ats], True
        # The other side of a cation-pi is the charged group. The
        # measurement was made from its heteroatom, so that is where the
        # line belongs - not the carbon that happens to be first.
        for a in ats:
            if str(ms.element[a]) not in ("C", "H"):
                return [xyz(a)], False
        return [xyz(ats[0])], False

    lines = []
    for it in res.interactions:
        if not it.lig_atoms or not it.rec_atoms:
            continue
        row = {
            "kind": it.kind,
            "quality": round(float(it.quality), 3),
            "distance": round(float(it.distance), 3),
            "label": it.label,
            "a": xyz(it.lig_atoms[0]),
            "b": xyz(it.rec_atoms[0]),
            "flags": list(it.flags or ()),
        }
        if it.kind in CENTROID_KINDS:
            a_pts, a_ring = endpoint(it.lig_atoms)
            b_pts, b_ring = endpoint(it.rec_atoms)
            row["a_span"] = a_pts
            row["b_span"] = b_pts
            row["spans"] = [bool(a_ring), bool(b_ring)]
        lines.append(row)

    clashes = []
    sr = res.strain
    if sr is not None:
        for c in (sr.clashes + sr.internal):
            clashes.append({
                "kind": c.kind,
                "overlap": round(float(c.overlap), 3),
                "distance": round(float(c.distance), 3),
                "severe": bool(c.severe),
                "label": c.label,
                "a": xyz(c.i), "b": xyz(c.j),
            })

    # Buried polar atoms with no partner, as positions like everything
    # else the page draws.
    stranded = [{"xyz": xyz(u.atom), "role": u.role, "side": u.side,
                 "label": u.label} for u in (res.unsatisfied or ())]

    return _json({"name": res.name, "lines": lines, "clashes": clashes,
                  "stranded": stranded,
                  "n_receptor": int(res.n_receptor),
                  "site": site_atoms(res)})


# How close a receptor atom has to be to the ligand to be worth drawing.
# 4.5 A is the usual "lining the pocket" cut: it takes the residues an
# interaction could plausibly come from without dragging in the second
# shell, which turns a pocket into a hairball.
SITE_RADIUS = 4.5


def site_atoms(res):
    """Receptor atoms lining the pocket, as coordinates.

    Chosen here rather than in the viewer for the same reason the
    interaction lines are: the browser's reader numbers atoms its own
    way, and anything decided there is something that can disagree with
    ChimeraX. The page turns these into whole residues and draws them.
    """
    import numpy as np
    ms = res.combined
    n_rec = int(res.n_receptor)
    if ms is None or n_rec <= 0 or n_rec >= ms.n_atoms:
        return []
    lig = ms.xyz[n_rec:]
    rec = ms.xyz[:n_rec]
    heavy = ms.element[n_rec:] != "H"
    if heavy.any():
        lig = lig[heavy]
    # one pass, chunked: a 2800-atom receptor against 40 ligand atoms is
    # small, but a big site would otherwise allocate n_rec x n_lig
    near = np.zeros(n_rec, dtype=bool)
    r2 = SITE_RADIUS * SITE_RADIUS
    for k in range(0, len(lig), 32):
        d = rec[:, None, :] - lig[None, k:k + 32, :]
        near |= (d * d).sum(axis=2).min(axis=1) <= r2
    out = []
    for i in np.flatnonzero(near):
        v = ms.xyz[int(i)]
        out.append([round(float(v[0]), 3), round(float(v[1]), 3),
                    round(float(v[2]), 3)])
    return out


# --------------------------------------------------------------- torsions
#
# The flagged torsions of the analysed pose, and the QM profile for one of
# them. Both already existed in the ChimeraX bundle and both were built
# from these same objects; the web page was simply never given them, so
# its panel showed an empty list where ChimeraX shows the strain.


def _xyz(ms, i):
    v = ms.xyz[int(i)]
    return [round(float(v[0]), 3), round(float(v[1]), 3), round(float(v[2]), 3)]


def torsions():
    """Every flagged torsion, worst first.

    Worst first means by measured cost where there is one. A torsion with
    no QM class has no number and cannot be ranked against one that has,
    so those come after, in bond order - the ordering the ChimeraX list
    uses, because the two lists are meant to be the same list.

    Each row carries the two bond atoms as coordinates as well as their
    indices. The page's viewer matches atoms on position, never on index,
    for the same reason the interaction endpoints do.
    """
    res = _STATE.get("last")
    if res is None or res.strain is None or res.combined is None:
        return _json({"rows": [], "warn_kcal": None})
    from hitpicking.strain.torsions import QM_WARN_KCAL
    ms = res.combined
    rows = []
    for t in res.strain.torsions:
        if not t.flagged:
            continue
        i, j = t.bond
        rows.append({
            "bond": "%d,%d" % (i, j),
            "label": " - ".join(ms.atom_label(a) for a in t.bond),
            "kind": t.subtype or t.bond_type,
            "angle": round(t.angle, 1),
            "assessment": t.assessment,
            "severity": t.severity,
            "energy": (None if t.energy is None else round(t.energy, 2)),
            "note": t.note,
            "atoms": [int(a) for a in t.atoms],
            "a": _xyz(ms, i),
            "b": _xyz(ms, j),
            # all four, in order: the pair is the bond, the outer two are
            # what the angle is measured between, and the viewer wants to
            # light up what is actually twisted rather than one bond.
            "xyz": [_xyz(ms, a) for a in t.atoms],
        })
    rows.sort(key=lambda r: (r["energy"] is None, -(r["energy"] or 0.0),
                             r["bond"]))
    return _json({"rows": rows, "warn_kcal": QM_WARN_KCAL})


def _atom_at(ms, x, y, z, tolerance=0.05):
    """The atom at this position, or None. Position, never index."""
    import numpy as np
    d = ms.xyz - np.array([float(x), float(y), float(z)], dtype=float)
    i = int(np.argmin((d * d).sum(axis=1)))
    if float(np.sqrt((d[i] * d[i]).sum())) > float(tolerance):
        return None
    return i


def atoms_xyz(indices):
    """Where these atoms are, for a page that works in coordinates.

    The panel's rows carry atom indices, because that is what the
    ChimeraX host selects with. The web page has no use for an index -
    its viewer numbers atoms its own way - so it asks where they are.
    """
    res = _STATE.get("last")
    if res is None or res.combined is None:
        return _json([])
    ms = res.combined
    out = []
    for i in indices or ():
        try:
            k = int(i)
        except (TypeError, ValueError):
            continue
        if 0 <= k < ms.n_atoms:
            out.append(_xyz(ms, k))
    return _json(out)


def torsion_at(ax, ay, az, bx, by, bz):
    """The torsion on the bond between these two atoms, if there is one.

    For clicking a bond in the viewer rather than a row in the list. Any
    rotatable bond the core measured answers here, not only the flagged
    ones: "this one is fine" is a useful answer about a bond you are
    looking at, and it is the same measurement either way.
    """
    res = _STATE.get("last")
    if res is None or res.strain is None or res.combined is None:
        return _json({"error": "analyse a pose first"})
    ms = res.combined
    i = _atom_at(ms, ax, ay, az)
    j = _atom_at(ms, bx, by, bz)
    if i is None or j is None:
        return _json({"error": "that bond is not in the analysis"})
    want = tuple(sorted((i, j)))
    for t in res.strain.torsions:
        if tuple(sorted(t.bond)) != want:
            continue
        return _json({
            "bond": "%d,%d" % t.bond,
            "label": " - ".join(ms.atom_label(a) for a in t.bond),
            "kind": t.subtype or t.bond_type,
            "angle": round(t.angle, 1),
            "assessment": t.assessment,
            "severity": t.severity,
            "flagged": bool(t.flagged),
            "energy": (None if t.energy is None else round(t.energy, 2)),
            "note": t.note,
            "atoms": [int(a) for a in t.atoms],
            "xyz": [_xyz(ms, a) for a in t.atoms],
            "profile": json.loads(torsion_profile("%d,%d" % t.bond)),
        })
    # Not every bond has one: ring bonds, double bonds and terminal bonds
    # are not rotatable, so there is nothing to measure. Name the bond -
    # a click that registered must not read as a click that did nothing.
    return _json({"error": "%s - %s is not a rotatable bond (ring, double "
                           "or terminal), so it has no torsion"
                           % (ms.atom_label(i), ms.atom_label(j))})


def torsion_profile(bond=None):
    """The QM profile for one bond, or for the worst flagged torsion.

    A torsion belongs to a bond, so the clicked row decides it. Returns
    null rather than raising when there is no QM class for that bond
    class: most bonds have none, and an empty plot is the honest answer.
    """
    res = _STATE.get("last")
    if res is None or res.strain is None:
        return _json(None)
    from hitpicking.strain import torsions as T
    if T.qm_reference() is None:
        return _json(None)
    ts = list(res.strain.torsions)
    if not ts:
        return _json(None)

    chosen = None
    if bond:
        try:
            want = tuple(sorted(int(x) for x in str(bond).split(",")))
        except (TypeError, ValueError):
            want = None
        if want is not None:
            for t in ts:
                if tuple(sorted(t.bond)) == want:
                    chosen = t
                    break
    if chosen is None:
        flagged = [t for t in ts if t.flagged]
        flagged.sort(key=lambda t: (t.energy is None, -(t.energy or 0.0)))
        chosen = flagged[0] if flagged else None
    if chosen is None:
        return _json(None)
    try:
        return _json(T.profile_view(chosen.bond_type, chosen.subtype,
                                    chosen.ortho_substituted, chosen.angle))
    except Exception:
        return _json(None)


# ------------------------------------------------------------------ click

def explain_at(x, y, z, tolerance=0.05):
    """The handbook entry for the atom at this position.

    The page clicks an atom in its own viewer and sends where it is, not
    which number it thinks it is. Matching on position keeps the two
    parsers from having to agree on ordering - and when the clicked atom
    is one the analysis did not include (a water the reader dropped, say)
    this says so rather than explaining the wrong atom.
    """
    import numpy as np
    res = _STATE.get("last")
    if res is None or res.combined is None:
        raise RuntimeError("analyse a pose first")
    ms = res.combined
    d = ms.xyz - np.array([float(x), float(y), float(z)], dtype=float)
    i = int(np.argmin((d * d).sum(axis=1)))
    off = float(np.sqrt((d[i] * d[i]).sum()))
    if off > float(tolerance):
        return _json({"error": "that atom is not in the analysis",
                      "nearest": round(off, 3)})
    return explain(None, i)


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

"""Which of a pose's own score terms the model is allowed to learn from.

The descriptor block ends with a few slots holding the docking program's
own numbers. Until now those slots were wired to DOCK 3's seven term names
and nothing else, so a mol2 from Glide, Vina, GOLD or rDock put zero in
every one of them: the model was blind to the score its user trusts most.

This module turns that hard-coded list into a choice. It does three jobs:

  1. looks at the numbers a pose actually carries and says which of them
     are worth feeding in and which are junk - a row number, a constant,
     a failed term that came through as a sentinel;
  2. recognises a few well-known programs by their term names, so the
     common cases need no thought from the user;
  3. normalises whatever the user picks into a fixed-width, ordered
     selection, because the slots are columns in a trained model and a
     model may only be used with the selection it was trained under.

The number of slots is deliberately NOT user-settable. Each is a column in
the descriptor block, and that block is variance-weighted against the two
fingerprint blocks as a whole: adding score columns silently dilutes the
clash count and the interaction counts, which are the features a grader
actually reasons about. Seven was DOCK's own count and is kept.

On the program table: only the DOCK entry is verified against real output
in this repository (`more_opcore_enum_asp_h2o_hbond.mol2`). The others are
name profiles - they order and label a selection when those names show up,
and nothing more. The generic path below is what actually has to work, and
it needs no table at all.
"""

from __future__ import annotations

import math

N_SCORE_SLOTS = 7

# Verified against DOCK 3 mol2 output read by this tool's own parser.
DOCK_TERMS = ("Total Energy", "Van der Waals", "Electrostatic",
              "Ligand Polar Desolv", "Ligand Apolar Desolv",
              "Total Strain", "Max Strain")

DEFAULT_TERMS = DOCK_TERMS


class Program:
    """One docking program's term names, in the order worth feeding in."""

    def __init__(self, pid, name, terms, verified=False, note=""):
        self.id = pid
        self.name = name
        self.terms = tuple(terms)
        self.verified = bool(verified)
        self.note = note

    def describe(self):
        return {"id": self.id, "name": self.name, "terms": list(self.terms),
                "verified": self.verified, "note": self.note}


PROGRAMS = (
    Program("dock", "DOCK 3", DOCK_TERMS, verified=True,
            note="read from the mol2 comment block, as ViewDock reads it"),
    Program("vina", "AutoDock Vina / smina / gnina",
            ("minimizedAffinity", "affinity", "CNNaffinity", "CNNscore",
             "CNN_VS", "minimizedRMSD"),
            note="smina and gnina write these as SD properties"),
    Program("glide", "Glide",
            ("r_i_docking_score", "r_i_glide_gscore", "r_i_glide_emodel",
             "r_i_glide_evdw", "r_i_glide_ecoul", "r_i_glide_lipo",
             "r_i_glide_hbond", "r_i_glide_einternal")),
    Program("gold", "GOLD",
            ("Gold.PLP.Fitness", "Gold.Goldscore.Fitness",
             "Gold.Chemscore.Fitness", "Gold.ASP.Fitness",
             "Gold.PLP.Chemscore.Hbond", "Gold.PLP.part.hbond")),
    Program("rdock", "rDock",
            ("SCORE", "SCORE.INTER", "SCORE.INTRA", "SCORE.RESTR",
             "SCORE.norm")),
    Program("autodock4", "AutoDock 4",
            ("Binding Energy", "Intermol Energy", "Internal Energy",
             "Torsional Energy", "Ligand Efficiency")),
)

PROGRAMS_BY_ID = {p.id: p for p in PROGRAMS}

# Names that are bookkeeping, not chemistry. Matched case-insensitively
# against the whole name, so "Rank" is excluded and "Rank Score" is not.
NOT_A_SCORE = frozenset((
    "number", "rank", "setnum", "matchnum", "cloud", "flexreccode",
    "id", "index", "entry", "pose", "conformer", "conf", "cluster",
    "cluster id", "model", "mol", "molecule", "serial", "row",
    "ligand source file", "protonation", "arbitrary",
))

# Words that make a name look like a score, used only to order the
# generic pick when more terms vary than there are slots.
SCORE_WORDS = ("score", "energy", "affinity", "fitness", "gscore", "emodel",
               "binding", "total", "inter", "vdw", "waals", "electro",
               "coul", "desolv", "strain", "hbond", "lipo", "polar",
               "chemgauss", "rmsd", "efficiency", "penalty")

SENTINEL = 500.0          # |value| beyond this is a failed term, not a number


def _clean_name(name):
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _score_likeness(name):
    c = _clean_name(name)
    return sum(1 for w in SCORE_WORDS if w in c)


def _looks_like_an_id(name, values):
    """A row number dressed as a float.

    DOCK writes Number, Rank, Setnum and Matchnum into the same comment
    block as the energies, all as floats. Feeding a row number to the model
    teaches it the order the file happened to be in.
    """
    if _clean_name(name) in NOT_A_SCORE:
        return True
    if not values:
        return False
    if _score_likeness(name) or not values:
        return False         # it says it is a score; take it at its word
    if not all(float(v).is_integer() for v in values):
        return False
    if all(v >= 0 for v in values) and len(set(values)) > max(2, 0.8 * len(values)):
        return True          # near-unique non-negative whole numbers: an index
    return False


def inspect(samples):
    """What each numeric term looks like across a set of poses.

    `samples` is an iterable of {term: value} dicts - one per pose, as
    PoseResult.scores gives them. Returns a list of dicts, one per term,
    ordered the way `suggest` would pick them.
    """
    samples = [s for s in samples if isinstance(s, dict)]
    seen = {}
    for s in samples:
        for k, v in s.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            v = float(v)
            if v != v or math.isinf(v):
                continue
            seen.setdefault(str(k), []).append(v)

    n_poses = max(len(samples), 1)
    rows = []
    for name, values in seen.items():
        live = [v for v in values if abs(v) <= SENTINEL]
        n_sentinel = len(values) - len(live)
        distinct = len(set(round(v, 9) for v in live))
        if live:
            mean = sum(live) / len(live)
            var = sum((v - mean) ** 2 for v in live) / len(live)
            sd = math.sqrt(var)
        else:
            mean = sd = 0.0
        if _looks_like_an_id(name, live):
            verdict, why = "identifier", "a row number, not a measurement"
        elif not live:
            verdict, why = ("unusable",
                            "every value is a sentinel (|v| > %g)" % SENTINEL)
        elif distinct < 2:
            verdict, why = ("constant",
                            "the same value on every pose, so it separates "
                            "nothing")
        elif n_sentinel > len(values) / 2:
            verdict, why = ("unusable",
                            "%d of %d poses carry a sentinel"
                            % (n_sentinel, len(values)))
        else:
            verdict, why = "varies", ""
        rows.append({
            "name": name,
            "n": len(values),
            "coverage": round(len(values) / float(n_poses), 3),
            "distinct": distinct,
            "mean": round(mean, 4),
            "sd": round(sd, 4),
            "sentinels": n_sentinel,
            "verdict": verdict,
            "why": why,
            "usable": verdict == "varies",
        })
    rows.sort(key=lambda r: (not r["usable"], -_score_likeness(r["name"]),
                             -r["sd"], r["name"]))
    return rows


def detect_program(names):
    """The program whose term names best match, or None.

    Returns (Program, matched_count). Matching is on exact names: a
    half-match is reported rather than hidden, because "this looks like
    Glide but only three of eight terms are here" is worth knowing.
    """
    have = set(str(n) for n in names)
    best, best_n = None, 0
    for p in PROGRAMS:
        n = sum(1 for t in p.terms if t in have)
        if n > best_n:
            best, best_n = p, n
    if best is None or best_n < 2:
        return None, 0
    return best, best_n


def suggest(samples, slots=N_SCORE_SLOTS):
    """Pick the terms to feed the model, and say why.

    Tries the known programs first - if a program's names are present, its
    own ordering is used, because it was chosen by people who know what the
    terms mean. Anything the program names that is constant or missing on
    this particular set is dropped, and the free slots are filled from
    whatever else varies.
    """
    rows = inspect(samples)
    by_name = {r["name"]: r for r in rows}
    usable = [r["name"] for r in rows if r["usable"]]

    program, matched = detect_program(by_name)
    chosen, source = [], "generic"
    if program is not None:
        chosen = [t for t in program.terms if by_name.get(t, {}).get("usable")]
        if chosen:
            source = program.id
    for name in usable:
        if len(chosen) >= slots:
            break
        if name not in chosen:
            chosen.append(name)
    chosen = chosen[:slots]

    if not chosen:
        why = ("no pose carries a usable score term, so the model will "
               "learn from geometry and interactions alone")
    elif source == "generic":
        why = ("no docking program was recognised by name; these are the "
               "terms that vary across the poses")
    else:
        why = "%s was recognised by name (%d of %d terms present)" % (
            program.name, matched, len(program.terms))

    return {"terms": tuple(chosen), "source": source,
            "program": program.describe() if program is not None else None,
            "matched": matched, "why": why, "available": rows,
            "slots": slots}


def normalize(terms, slots=N_SCORE_SLOTS):
    """A selection in canonical form: strings, no duplicates, at most
    `slots` of them, order preserved."""
    out = []
    for t in (terms or ()):
        t = str(t).strip()
        if t and t not in out:
            out.append(t)
        if len(out) >= slots:
            break
    return tuple(out)


def signature(terms):
    """What a model records so it cannot be used with a different pick.

    Position matters - the terms are columns - so this is the ordered
    tuple, not a set.
    """
    return list(normalize(terms))

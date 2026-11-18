"""Ride alongside ChimeraX's ViewDock tool.

ViewDock has no public API for "the current pose changed". The only hook is
the Qt signal on its ItemTable, and its per-pose dictionary. Both are
internals, so EVERY access here is wrapped: if ViewDock changes shape in a
later release the bundle must lose this integration, not break.

THE CONTRACT WITH VIEWDOCK
--------------------------
ViewDock's update_model_description() does, for every pose:

    column_map = { col.title: col for col in self.struct_table.columns }
    all_titles = [self.display_key(key) for key in structure.viewdock_data]
    ... column_map[title] ...

so **every key in viewdock_data must have a column whose title is exactly
display_key(key)**, or ViewDock raises KeyError on every row click. Its own
process_hbonds obeys this by using the key "H-Bonds" and the column title
"H-Bonds".

Two rules follow, and both are enforced below:
  1. our keys are chosen so display_key(key) == key (no underscores or dots),
     and the column title IS the key;
  2. a key is never written into viewdock_data unless its column exists -
     and if a column cannot be created, the key is removed again.

What we add:
  * analyse each pose as the user steps through the table
  * extra columns with our interaction counts and a geometry-quality total
  * ratings read back out for the active-learning module
"""

from __future__ import annotations

_HOOKS = {}

# Column titles. These double as the viewdock_data keys: see the contract
# above. Deliberately free of underscores and dots so display_key() is the
# identity, and prefixed so they cannot collide with a docking program's own
# descriptors.
COL_HBONDS = "HP H-Bonds"
COL_SALT = "HP Salt"
COL_STACK = "HP Stack"
COL_SCORE = "HP Quality"
COL_CLASH = "HP Clash"
COLUMNS = (COL_HBONDS, COL_SALT, COL_STACK, COL_SCORE, COL_CLASH)

# Keys written by earlier builds of this bundle. They used underscores, so
# ViewDock's display_key turned "hitpick_hbonds" into "hitpick hbonds", which
# matched no column and raised KeyError on every row click. Any pose still
# carrying them - from a saved session, or a ChimeraX left running across an
# upgrade - is cleaned up on hook.
LEGACY_KEYS = ("hitpick_hbonds", "hitpick_salt", "hitpick_stack",
               "hitpick_score", "HP HBonds", "HP Score")

RATING_KEY = "rating"
# ViewDock's scale, lowest first. Matches AutoParty's 3-level ordinal grading.
RATING_LABELS = ("(unrated)", "bad", "maybe", "good")


def display_key_identity(key):
    """Mirror of ViewDock.display_key, used to assert our invariant."""
    out = []
    for space_word in key.split():
        for fragment in space_word.split("_"):
            if fragment.endswith("."):
                out.append(fragment)
            else:
                out.extend(fragment.split("."))
    return " ".join(out)


# Fail fast at import if a future edit breaks the contract.
for _c in COLUMNS:
    assert display_key_identity(_c) == _c, (
        "viewdock column %r would be displayed as %r; ViewDock would raise "
        "KeyError on every row click" % (_c, display_key_identity(_c)))


def find_viewdock(session):
    """The running ViewDock tool instance, or None."""
    try:
        from chimerax.viewdock.tool import ViewDockTool
    except Exception:
        return None
    try:
        found = session.tools.find_by_class(ViewDockTool)
    except Exception:
        return None
    return found[0] if found else None


def autohook(session):
    """Hook ViewDock if it is already running. Safe to call repeatedly."""
    tool = find_viewdock(session)
    if tool is None:
        return False
    return hook(session, tool)


def hook(session, tool=None):
    tool = tool or find_viewdock(session)
    if tool is None:
        return False
    if session in _HOOKS:
        return True
    table = getattr(tool, "struct_table", None)
    sig = getattr(table, "selection_changed", None)
    if sig is None:
        session.logger.info(
            "hitpick: this ViewDock has no selection_changed signal; "
            "stepping through poses will not auto-analyse.")
        return False

    def on_selection(selected, deselected):
        try:
            _on_pose_changed(session, selected)
        except Exception as e:
            session.logger.warning("hitpick: pose change failed: %s" % e)

    try:
        sig.connect(on_selection)
    except Exception as e:
        session.logger.info("hitpick: could not connect to ViewDock (%s)" % e)
        return False

    _HOOKS[session] = (tool, on_selection)
    # A previous session may have left our keys on the structures while this
    # table has no such columns yet; reconcile before anything clicks a row.
    _reconcile(session, tool)
    session.logger.info("hitpick: linked to ViewDock; poses will be analysed "
                        "as you select them.")
    return True


def unhook(session, strip_data=True):
    rec = _HOOKS.pop(session, None)
    if not rec:
        return False
    tool, cb = rec
    try:
        tool.struct_table.selection_changed.disconnect(cb)
    except Exception:
        pass
    if strip_data:
        _strip_keys(tool)
    return True


def is_hooked(session):
    return session in _HOOKS


def _on_pose_changed(session, selected):
    """Analyse the newly selected pose.

    Guarded against re-entry: both analysing (which adds pseudobond groups)
    and writing our columns can make ViewDock emit selection_changed again,
    and this handler would then call itself until Python gave up. Guarding
    only hitpick_analyze is not enough, because update_columns can trigger
    the signal on its own.
    """
    if not selected:
        return
    if getattr(session, "_hitpick_in_pose_change", False):
        return
    from .cmd import hitpick_analyze
    from .state import get_state
    st = get_state(session)
    if not st.ready:
        return
    structure = selected[0]
    if structure is st.receptor_model:
        return
    session._hitpick_in_pose_change = True
    try:
        hitpick_analyze(session, structure)
        update_columns(session, structure)
    finally:
        session._hitpick_in_pose_change = False


# --------------------------------------------------------------- columns

def _structures(tool):
    try:
        return list(tool.structures)
    except Exception:
        return []


def _existing_titles(tool):
    try:
        return {col.title: col for col in tool.struct_table.columns}
    except Exception:
        return {}


def _ensure_columns(session, tool):
    """Create any of our columns that are missing. Returns those that exist.

    A column is only reported as available once it is really in the table, so
    callers never write a key ViewDock cannot resolve.
    """
    present = _existing_titles(tool)
    ok = set()
    for title in COLUMNS:
        if title in present:
            ok.add(title)
            continue
        try:
            tool.struct_table.add_column(
                title, lambda s, k=title: _cell(s, k))
        except Exception as e:
            session.logger.info(
                "hitpick: could not add the %r column to ViewDock (%s); "
                "its values will not be shown there." % (title, e))
            continue
        if title in _existing_titles(tool):
            ok.add(title)
    return ok


def _cell(structure, key):
    data = getattr(structure, "viewdock_data", None) or {}
    v = data.get(key, "")
    return "" if v is None else v


def _strip_keys(tool, titles=COLUMNS):
    """Remove our keys from every pose. Used when we cannot back them up
    with columns, and on unhook."""
    n = 0
    for s in _structures(tool):
        data = getattr(s, "viewdock_data", None)
        if not isinstance(data, dict):
            continue
        for t in titles:
            if data.pop(t, None) is not None:
                n += 1
    return n


def _reconcile(session, tool):
    """Make the table consistent with whatever keys the poses already carry.

    ViewDock can be closed and reopened, giving a fresh table while the
    structures keep their viewdock_data. Any of our keys left over would then
    have no column, and the next row click would raise.
    """
    # Anything an older build left behind can never resolve: drop it first.
    legacy = _strip_keys(tool, LEGACY_KEYS)
    if legacy:
        session.logger.info(
            "hitpick: cleared %d value(s) left by an earlier version that "
            "ViewDock could not resolve" % legacy)

    stale = set()
    for s in _structures(tool):
        data = getattr(s, "viewdock_data", None) or {}
        stale |= {t for t in COLUMNS if t in data}
    if not stale:
        return
    ok = _ensure_columns(session, tool)
    missing = stale - ok
    if missing:
        removed = _strip_keys(tool, missing)
        session.logger.info(
            "hitpick: removed %d stale value(s) ViewDock had no column for"
            % removed)


def annotate_structure(session, structure, result, allowed=None):
    """Write our per-pose numbers into ViewDock's own data dict.

    `allowed` is the set of column titles known to exist. Anything outside it
    is not written, and is cleared if it is already there.
    """
    data = getattr(structure, "viewdock_data", None)
    if not isinstance(data, dict):
        try:
            structure.viewdock_data = data = {}
        except Exception:
            return False
    c = result.counts
    values = {
        COL_HBONDS: c.get("hbond", 0) + c.get("hbond_charged", 0),
        COL_SALT: c.get("salt_bridge", 0),
        COL_STACK: (c.get("pi_stack_parallel", 0)
                    + c.get("pi_stack_tshaped", 0)),
        # Sum of geometric qualities: a crude "how much well-formed
        # interaction is there". Explicitly NOT an energy, never shown as one.
        COL_SCORE: round(sum(i.quality for i in result.interactions), 2),
        # worst vdW overlap in Angstrom; blank when strain is switched off
        COL_CLASH: (round(result.strain.worst_overlap, 2)
                    if getattr(result, "strain", None) is not None else ""),
    }
    for title, v in values.items():
        if allowed is not None and title not in allowed:
            data.pop(title, None)
            continue
        data[title] = v
    return True


def update_columns(session, structure=None):
    """Refresh our columns in the ViewDock table.

    Columns are created BEFORE any value is written, so viewdock_data can
    never hold a key the table cannot resolve.
    """
    tool = find_viewdock(session)
    if tool is None:
        return False
    allowed = _ensure_columns(session, tool)

    from .state import get_state
    st = get_state(session)
    if st.result is not None and structure is not None:
        annotate_structure(session, structure, st.result, allowed=allowed)

    present = _existing_titles(tool)
    for title in COLUMNS:
        col = present.get(title)
        if col is None:
            continue
        try:
            tool.struct_table.update_column(col, data=True)
        except Exception:
            continue
    return True


# ---------------------------------------------------------------- ratings

def iter_rated(session):
    """(structure, rating_int) for every pose ViewDock knows about."""
    tool = find_viewdock(session)
    if tool is None:
        return []
    out = []
    for s in _structures(tool):
        data = getattr(s, "viewdock_data", None) or {}
        r = data.get(RATING_KEY, 0)
        try:
            r = int(r)
        except Exception:
            r = 0
        out.append((s, r))
    return out


def rated_poses(session):
    """Only the poses a human has actually graded."""
    return [(s, r) for s, r in iter_rated(session) if r > 0]


def rating_label(value):
    try:
        return RATING_LABELS[int(value)]
    except Exception:
        return RATING_LABELS[0]


def set_rating(session, structure, value):
    data = getattr(structure, "viewdock_data", None)
    if not isinstance(data, dict):
        return False
    data[RATING_KEY] = int(value)
    tool = find_viewdock(session)
    if tool is not None:
        try:
            tool.struct_table.update_cell("Rating", structure)
        except Exception:
            pass
    return True

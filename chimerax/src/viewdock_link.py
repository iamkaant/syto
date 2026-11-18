"""Ride alongside ChimeraX's ViewDock tool.

ViewDock has no public API for "the current pose changed". The only hook is
the Qt signal on its ItemTable, and its per-pose dictionary. Both are
internals, so EVERY access here is wrapped: if ViewDock changes shape in a
later release the bundle must lose this integration, not break.

What we add:
  * analyse each pose as the user steps through the table
  * extra columns (H-bonds, salt bridges, stacking, a quality sum) written
    into viewdock_data, exactly as ViewDock's own process_hbonds does
  * ratings read back out for the active-learning module
"""

from __future__ import annotations

_HOOKS = {}

# Columns we add. Keys are what we write into structure.viewdock_data.
COLUMNS = (
    ("HP HBonds", "hitpick_hbonds"),
    ("HP Salt", "hitpick_salt"),
    ("HP Stack", "hitpick_stack"),
    ("HP Score", "hitpick_score"),
)

RATING_KEY = "rating"
# ViewDock's scale, lowest first. Matches AutoParty's 3-level ordinal grading.
RATING_LABELS = ("(unrated)", "bad", "maybe", "good")


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
    session.logger.info("hitpick: linked to ViewDock; poses will be analysed "
                        "as you select them.")
    return True


def unhook(session):
    rec = _HOOKS.pop(session, None)
    if not rec:
        return False
    tool, cb = rec
    try:
        tool.struct_table.selection_changed.disconnect(cb)
    except Exception:
        pass
    return True


def is_hooked(session):
    return session in _HOOKS


def _on_pose_changed(session, selected):
    if not selected:
        return
    from .cmd import hitpick_analyze
    from .state import get_state
    st = get_state(session)
    if not st.ready:
        return
    structure = selected[0]
    if structure is st.receptor_model:
        return
    hitpick_analyze(session, structure)
    update_columns(session, structure)


# --------------------------------------------------------------- columns

def annotate_structure(session, structure, result):
    """Write our per-pose numbers into ViewDock's own data dict."""
    data = getattr(structure, "viewdock_data", None)
    if data is None:
        try:
            structure.viewdock_data = data = {}
        except Exception:
            return False
    c = result.counts
    data["hitpick_hbonds"] = c.get("hbond", 0) + c.get("hbond_charged", 0)
    data["hitpick_salt"] = c.get("salt_bridge", 0)
    data["hitpick_stack"] = (c.get("pi_stack_parallel", 0)
                             + c.get("pi_stack_tshaped", 0))
    # Sum of geometric qualities: a crude "how much well-formed interaction
    # is there". Explicitly NOT an energy, and never shown as one.
    data["hitpick_score"] = round(
        sum(i.quality for i in result.interactions), 2)
    return True


def update_columns(session, structure=None):
    """Add or refresh our columns in the ViewDock table."""
    tool = find_viewdock(session)
    if tool is None:
        return False
    from .state import get_state
    st = get_state(session)
    if st.result is not None and structure is not None:
        annotate_structure(session, structure, st.result)

    table = getattr(tool, "struct_table", None)
    if table is None:
        return False
    existing = {}
    try:
        for col in table.columns:
            existing[col.title] = col
    except Exception:
        return False

    for title, key in COLUMNS:
        col = existing.get(title)
        try:
            if col is None:
                table.add_column(title,
                                 lambda s, k=key: _cell(s, k),
                                 format="%s")
            else:
                table.update_column(col, data=True)
        except Exception:
            continue
    return True


def _cell(structure, key):
    data = getattr(structure, "viewdock_data", None) or {}
    v = data.get(key, "")
    return "" if v is None else v


# ---------------------------------------------------------------- ratings

def iter_rated(session):
    """(structure, rating_int) for every pose ViewDock knows about."""
    tool = find_viewdock(session)
    if tool is None:
        return []
    out = []
    try:
        structures = list(tool.structures)
    except Exception:
        return []
    for s in structures:
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
    if data is None:
        return False
    data[RATING_KEY] = int(value)
    tool = find_viewdock(session)
    if tool is not None:
        try:
            tool.struct_table.update_cell("Rating", structure)
        except Exception:
            pass
    return True

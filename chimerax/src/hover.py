"""Replace ChimeraX's hover balloon with the handbook's view of what is under
the cursor.

ChimeraX's ObjectIdMouseMode shows `pick.description()` in a QLabel balloon
and then fires the 'mouse hover' trigger with that pick. Because the balloon
is a QLabel it accepts HTML, and because the trigger fires AFTER the text is
set, a handler can simply replace the text in place - no mouse mode to
install, no binding for the user to set, and the native behaviour returns
untouched the moment this is switched off.

Hovering an atom gives its role and its basicity; hovering one of our
interaction lines gives the measured geometry against its limits, which is
the question being asked when someone points at a dashed line.
"""

from __future__ import annotations

HOVER_TRIGGER = "mouse hover"
_HANDLERS = {}

# Colours matching the label bands in render.py
_BAND_HTML = {"good": "#1e8c46", "marginal": "#c88214", "edge": "#c83228"}


def _popup(session):
    try:
        return session.ui.main_window.graphics_window.popup
    except Exception:
        return None


def install(session):
    """Start replacing the balloon. Safe to call repeatedly."""
    if session in _HANDLERS:
        return True
    if not session.triggers.has_trigger(HOVER_TRIGGER):
        # Created by ObjectIdMouseMode. Adding it ourselves is what broke the
        # Toolbar once already, so we wait for ChimeraX instead.
        return False

    def on_hover(trigger_name, pick):
        try:
            _update_balloon(session, pick)
        except Exception:
            pass          # never let a tooltip break the hover

    _HANDLERS[session] = session.triggers.add_handler(HOVER_TRIGGER, on_hover)
    return True


def remove(session):
    h = _HANDLERS.pop(session, None)
    if h is None:
        return False
    try:
        session.triggers.remove_handler(h)
    except Exception:
        pass
    return True


def is_installed(session):
    return session in _HANDLERS


def _update_balloon(session, pick):
    from .state import get_state
    st = get_state(session)
    if not st.options.hover_detail:
        return
    popup = _popup(session)
    if popup is None:
        return
    html = describe(session, st, pick)
    if html:
        popup.setText(html)
        popup.adjustSize()


# ----------------------------------------------------------------- content

def _row(label, value, colour=None):
    v = ("<b><span style='color:%s'>%s</span></b>" % (colour, value)) \
        if colour else "<b>%s</b>" % value
    return "<tr><td style='padding-right:8px;color:#777'>%s</td>" \
           "<td>%s</td></tr>" % (label, v)


def describe(session, st, pick):
    """HTML for whatever is under the cursor, or None to leave it alone."""
    atom = getattr(pick, "atom", None)
    if atom is not None:
        return _describe_atom(st, atom)
    pb = getattr(pick, "pseudobond", None)
    if pb is not None:
        return _describe_pseudobond(st, pb)
    bond = getattr(pick, "bond", None)
    if bond is not None:
        return _describe_bond(st, bond)
    return None


def _describe_atom(st, atom):
    if st.molsys is None or st.handbook is None:
        return None
    idx = st.atom_index(atom)
    if idx is None:
        return None
    hit = st.handbook.for_atom(st.molsys, idx)
    rows = []
    title = "%s" % atom
    if hit and hit.title:
        # the title already carries the role; keep the atom spec short
        role = hit.title.split(" - ", 1)[-1]
        rows.append(_row("role", role))
    for a in (hit.annotations if hit else [])[:3]:
        rep = a.get("representative")
        if rep and isinstance(rep.get("value"), (int, float)):
            val = "%.2f <span style='color:#777'>(%s)</span>" % (
                float(rep["value"]), rep.get("compound", "parent"))
        else:
            val = "%.2f <span style='color:#777'>median</span>" % a["median"]
        val += " <span style='color:#777'>&nbsp;family %.1f to %.1f, n=%d</span>" \
               % (a["min"], a["max"], a["n"])
        if a.get("n_flagged"):
            val += " <span style='color:#c88214'>*</span>"
        rows.append(_row(a["quantity"], val))
    if hit and hit.warnings:
        rows.append(_row("", "<span style='color:#c88214'>%s</span>"
                         % hit.warnings[0]))
    if not rows:
        return None
    return ("<div style='font-family:-apple-system,sans-serif;font-size:11px'>"
            "<b>%s</b><table style='font-size:11px'>%s</table></div>"
            % (title, "".join(rows)))


def _find_interaction(st, pb):
    """The Interaction a drawn pseudobond came from."""
    if st.result is None:
        return None
    kind = getattr(pb, "hitpick_kind", None)
    if kind is None:
        return None
    best = None
    try:
        length = pb.length
    except Exception:
        return None
    for it in st.result.interactions:
        if it.kind != kind:
            continue
        # stacking lines are drawn between representative atoms, so the drawn
        # length is not the scored distance; match on the drawn atoms instead
        d = abs(it.distance - length)
        if best is None or d < best[0]:
            best = (d, it)
    return best[1] if best else None


def _describe_pseudobond(st, pb):
    kind = getattr(pb, "hitpick_kind", None)
    if kind is None:
        return None

    if kind in ("clash", "severe_clash", "internal_clash"):
        return _describe_clash(st, pb, kind)

    it = _find_interaction(st, pb)
    if it is None:
        return None
    from . import render
    band = render.interaction_band(it)
    colour = _BAND_HTML.get(band, "#333")

    rows = []
    limits = it.limits
    comps = it.components or {}

    def term(name, value, unit, key):
        lim = limits.get(key, "")
        q = comps.get(key)
        col = _BAND_HTML.get(render.quality_band(q), "#333") if q is not None else None
        v = "%s%s" % (value, unit)
        if lim:
            v += " <span style='color:#777'>limits %s</span>" % lim
        rows.append(_row(name, v, col))

    measure = getattr(it.criterion, "distance_measure", "atoms")
    dist_label = {"interplanar_separation": "plane separation",
                  "centroid": "centroid distance"}.get(measure, "distance")
    term(dist_label, "%.2f" % it.distance, " &Aring;", "distance")

    # `angles` also carries two LENGTHS for stacking - the lateral offset and
    # the centroid distance - so they must not be printed in degrees.
    LENGTHS = {"offset": "lateral offset", "centroid": "centroid distance"}
    for k, v in (it.angles or {}).items():
        if v is None or v != v:
            continue
        if k in LENGTHS:
            q = comps.get(k)
            from . import render as _r
            col = (_BAND_HTML.get(_r.quality_band(q)) if q is not None else None)
            extra = ""
            if k == "offset":
                from hitpicking.nci import criteria as _c
                extra = (" <span style='color:#777'>best near %.1f, up to "
                         "%.1f</span>" % (_c.PI_STACK_OFFSET[1],
                                          _c.PI_STACK_OFFSET[2]))
            rows.append(_row(LENGTHS[k], "%.2f &Aring;%s" % (v, extra), col))
            continue
        term(k, "%.0f" % v, "&deg;", k)

    rows.append(_row("atoms", it.label))
    if it.moiety_ids:
        rows.append(_row("moieties", " / ".join(
            str(m).replace("_", " ") for m in it.moiety_ids if m)))
    e = it.energy_estimate
    if e:
        rows.append(_row("estimate", "%.1f to %.1f kcal/mol "
                         "<span style='color:#777'>heuristic, not a free "
                         "energy</span>" % (e[0], e[2])))
    for f in it.flags[:2]:
        rows.append(_row("", "<span style='color:#c88214'>%s</span>" % f))

    header = "%s <span style='color:%s'>&mdash; %s</span>" % (
        it.criterion.label, colour,
        {"good": "within limits", "marginal": "off ideal",
         "edge": "only just inside"}.get(band, band))
    return ("<div style='font-family:-apple-system,sans-serif;font-size:11px'>"
            "<b>%s</b><table style='font-size:11px'>%s</table></div>"
            % (header, "".join(rows)))


def _describe_clash(st, pb, kind):
    if st.result is None or st.result.strain is None:
        return None
    try:
        length = pb.length
    except Exception:
        return None
    pool = (st.result.strain.internal if kind == "internal_clash"
            else st.result.strain.clashes)
    best = min(pool, key=lambda c: abs(c.distance - length), default=None)
    if best is None:
        return None
    rows = [
        _row("distance", "%.2f &Aring;" % best.distance),
        _row("overlap", "%.2f &Aring;" % best.overlap, "#c83228"),
        _row("atoms", best.label),
        _row("", "<span style='color:#777'>vdW overlap beyond %.1f &Aring; "
             "counts as a clash</span>" % 0.6),
    ]
    title = "Severe clash" if best.severe else (
        "Internal clash" if kind == "internal_clash" else "Clash")
    return ("<div style='font-family:-apple-system,sans-serif;font-size:11px'>"
            "<b style='color:#c83228'>%s</b>"
            "<table style='font-size:11px'>%s</table></div>"
            % (title, "".join(rows)))


def _describe_bond(st, bond):
    """A covalent bond: show its torsion if we judged one across it."""
    if st.result is None or st.result.strain is None or st.molsys is None:
        return None
    try:
        a1, a2 = bond.atoms
    except Exception:
        return None
    i1, i2 = st.atom_index(a1), st.atom_index(a2)
    if i1 is None or i2 is None:
        return None
    key = tuple(sorted((i1, i2)))
    for t in st.result.strain.torsions:
        if tuple(sorted(t.bond)) != key:
            continue
        colour = "#c83228" if t.flagged else "#777"
        rows = [
            _row("torsion", "%.0f&deg;" % t.angle, colour),
            _row("type", t.subtype or t.bond_type),
        ]
        if t.ortho_substituted:
            rows.append(_row("ortho", "yes &mdash; a twist here is expected"))
        if t.note:
            rows.append(_row("", "<span style='color:%s'>%s</span>"
                             % (colour, t.note)))
        return ("<div style='font-family:-apple-system,sans-serif;"
                "font-size:11px'><b>%s</b>"
                "<table style='font-size:11px'>%s</table></div>"
                % ("Rotatable bond", "".join(rows)))
    return None

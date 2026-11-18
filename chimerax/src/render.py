"""Drawing interactions, labels and highlights in the ChimeraX scene.

Each interaction type gets its own pseudobond group so the user can toggle
types independently from the Model Panel as well as from our own UI.

Colour encodes TYPE; opacity and radius encode QUALITY. A weak, badly aligned
hydrogen bond is drawn thin and faint, a textbook one thick and solid, so the
geometry is readable without opening a panel.
"""

from __future__ import annotations

GROUP_PREFIX = "hitpick "

# (r, g, b) 0-255. Chosen to stay distinguishable in both light and dark
# backgrounds and to survive the usual red/green colour-vision deficiencies.
TYPE_COLORS = {
    "hbond":             (90, 160, 255),
    "hbond_charged":     (40, 110, 230),
    "weak_hbond":        (150, 190, 230),
    "salt_bridge":       (235, 110, 60),
    "pi_stack_parallel": (190, 120, 230),
    "pi_stack_tshaped":  (150, 100, 200),
    "cation_pi":         (240, 180, 60),
    "halogen_bond":      (60, 200, 170),
    "hydrophobic":       (170, 170, 170),
    # Clashes are drawn too: a number in a log is no use when you are looking
    # at the pose. Red for a bad contact, deeper red when it is severe, and a
    # separate colour for the ligand clashing with itself.
    "clash":             (230, 70, 70),
    "severe_clash":      (190, 20, 20),
    "internal_clash":    (235, 130, 40),
}

TYPE_DASHES = {
    "hbond": 6, "hbond_charged": 6, "weak_hbond": 10,
    "salt_bridge": 4, "pi_stack_parallel": 8, "pi_stack_tshaped": 8,
    "cation_pi": 8, "halogen_bond": 6, "hydrophobic": 12,
    # solid, so a clash reads differently from a favourable interaction at a
    # glance rather than only by colour
    "clash": 0, "severe_clash": 0, "internal_clash": 0,
}

CLASH_KINDS = ("clash", "severe_clash", "internal_clash")

HIGHLIGHT_COLOR = (255, 230, 80, 255)


def group_name(kind):
    return GROUP_PREFIX + kind.replace("_", " ")


def clear_groups(session, kinds=None):
    """Remove our pseudobond groups, leaving everything else alone."""
    from chimerax.atomic import PseudobondGroup
    removed = []
    for m in list(session.models):
        if isinstance(m, PseudobondGroup) and m.name.startswith(GROUP_PREFIX):
            if kinds is None or m.name in {group_name(k) for k in kinds}:
                removed.append(m)
    if removed:
        session.models.close(removed)
    return len(removed)


def _quality_style(quality, kind):
    """Radius and alpha from the geometric quality."""
    q = max(0.0, min(1.0, float(quality)))
    radius = 0.06 + 0.12 * q
    alpha = int(90 + 165 * q)
    r, g, b = TYPE_COLORS.get(kind, (200, 200, 200))
    return radius, (r, g, b, alpha)


def draw_interactions(session, interactions, all_atoms, show_lengths=True,
                      show_angles=False, min_quality=0.0, show_limits=False):
    """Draw each interaction as a pseudobond, grouped by type.

    Ring-centroid interactions (stacking, cation-pi) have no single atom to
    attach to, so they are drawn to the ring atom nearest the partner. This is
    a rendering compromise and is noted in the label rather than hidden.
    """
    from chimerax.atomic import Atoms
    import numpy as np

    clear_groups(session)
    by_kind = {}
    for it in interactions:
        if it.quality < min_quality:
            continue
        by_kind.setdefault(it.kind, []).append(it)

    made = {}
    for kind, items in by_kind.items():
        name = group_name(kind)
        pbg = session.pb_manager.get_group(name, create=True)
        pbg.dashes = TYPE_DASHES.get(kind, 6)
        for it in items:
            a1 = _representative_atom(it.lig_atoms, it.rec_atoms, all_atoms)
            a2 = _representative_atom(it.rec_atoms, it.lig_atoms, all_atoms)
            if a1 is None or a2 is None or a1 is a2:
                continue
            try:
                pb = pbg.new_pseudobond(a1, a2)
            except Exception:
                continue
            radius, rgba = _quality_style(it.quality, kind)
            pb.radius = radius
            pb.color = rgba
            try:
                pb._hitpick_text = interaction_label_text(
                    it, show_lengths, show_angles, show_limits)
                pb._hitpick_band = interaction_band(it)
                pb.hitpick_kind = kind
            except Exception:
                pass
        if pbg.num_pseudobonds == 0:
            session.models.close([pbg])
        else:
            if pbg not in session.models:
                session.models.add([pbg])
            made[kind] = pbg.num_pseudobonds
    return made


def _representative_atom(group, other_group, all_atoms):
    """The atom of `group` closest to `other_group`, for drawing a line."""
    import numpy as np
    if not group:
        return None
    try:
        atoms = [all_atoms[int(i)] for i in group if 0 <= int(i) < len(all_atoms)]
        others = [all_atoms[int(i)] for i in other_group
                  if 0 <= int(i) < len(all_atoms)]
    except Exception:
        return None
    if not atoms:
        return None
    if len(atoms) == 1 or not others:
        return atoms[0]
    oc = np.mean([a.scene_coord for a in others], axis=0)
    best, bd = None, None
    for a in atoms:
        d = float(np.linalg.norm(a.scene_coord - oc))
        if bd is None or d < bd:
            best, bd = a, d
    return best


LABEL_ATTR = "hitpick_label"
_ATTR_REGISTERED = set()


def _register_attrs(session):
    """Register the per-object label attribute once per session.

    Labelling through the `attribute=` argument lets every atom or
    pseudobond carry its own text in a single call. Building one `label`
    COMMAND per atom from a spec string is what the earlier version did, and
    it failed silently for ligands with no chain id - which is why clicking
    "Label pKBHX" appeared to delete every label instead of adding any.
    """
    if session in _ATTR_REGISTERED:
        return
    from chimerax.atomic import Atom, Pseudobond
    for cls in (Atom, Pseudobond):
        try:
            cls.register_attr(session, LABEL_ATTR, "HitPickingTool",
                              attr_type=str)
        except Exception:
            pass          # already registered by an earlier session
    _ATTR_REGISTERED.add(session)


# How a geometric term is doing against its literature envelope. The score
# is already 1.0 at the ideal and 0 at the limit, so these are just names for
# bands of it.
GOOD, MARGINAL, EDGE = "good", "marginal", "edge"
BAND_COLORS = {
    GOOD:     (30, 140, 70),      # comfortably inside the limits
    MARGINAL: (200, 130, 20),     # inside, but well off ideal
    EDGE:     (200, 50, 40),      # only just inside; treat with suspicion
}
BAND_CUTOFFS = (0.55, 0.25)       # >=0.55 good, >=0.25 marginal, else edge


def quality_band(value):
    if value >= BAND_CUTOFFS[0]:
        return GOOD
    if value >= BAND_CUTOFFS[1]:
        return MARGINAL
    return EDGE


def interaction_band(it):
    """The band of the WORST term, which is what decides whether to trust it.

    A hydrogen bond at a perfect distance but 125 degrees off linear is not a
    good hydrogen bond, and averaging would hide that.
    """
    comps = [v for v in it.components.values() if v is not None]
    return quality_band(min(comps)) if comps else quality_band(it.quality)


def _fmt_range(lo, hi):
    return "%g-%g" % (round(lo, 2), round(hi, 2))


def interaction_label_text(it, show_lengths=True, show_angles=False,
                           show_limits=False, show_quality=False):
    """The text drawn on one interaction line.

    Deliberately terse. Colour carries whether the geometry is within its
    limits, so the text does not have to; `show_limits` adds the envelope in
    brackets for anyone who wants the numbers on screen.
    """
    crit = it.criterion
    parts = []
    if show_lengths:
        if show_limits:
            parts.append("%.2f[%s]" % (it.distance,
                                       _fmt_range(crit.d_min, crit.d_max)))
        else:
            parts.append("%.2f" % it.distance)
    if show_angles:
        for k, v in it.angles.items():
            if v is None or v != v:          # NaN: not measurable
                continue
            name = _short_angle(k)
            if show_limits and k in crit.angles:
                limit, ideal = crit.angles[k]
                bound = ("%g+" % limit) if ideal >= limit else ("<%g" % limit)
                parts.append("%s%.0f[%s]" % (name, v, bound))
            else:
                parts.append("%s%.0f" % (name, v))
    if show_quality:
        parts.append("q%.2f" % it.quality)
    return " ".join(parts)


def _short_angle(name):
    """Abbreviations kept very short: these are drawn over the structure."""
    return {"dha": "DHA", "haa": "HAX", "interplanar": "tilt",
            "offset": "off", "normal_offset": "off",
            "cxa": "CXA", "xay": "XAY"}.get(name, name)


def draw_clashes(session, strain, all_atoms, show_overlap=True):
    """Draw each clash as a solid pseudobond, grouped by severity.

    Separate groups so the Model Panel can toggle severe clashes, ordinary
    clashes and the ligand's internal clashes independently.
    """
    clear_groups(session, CLASH_KINDS)
    if strain is None:
        return {}

    buckets = {"clash": [], "severe_clash": [], "internal_clash": []}
    for c in strain.clashes:
        buckets["severe_clash" if c.severe else "clash"].append(c)
    for c in strain.internal:
        buckets["internal_clash"].append(c)

    _register_attrs(session)
    made = {}
    for kind, items in buckets.items():
        if not items:
            continue
        pbg = session.pb_manager.get_group(group_name(kind), create=True)
        pbg.dashes = TYPE_DASHES.get(kind, 0)
        r, g, b = TYPE_COLORS[kind]
        for c in items:
            if not (0 <= c.i < len(all_atoms) and 0 <= c.j < len(all_atoms)):
                continue
            try:
                pb = pbg.new_pseudobond(all_atoms[c.i], all_atoms[c.j])
            except Exception:
                continue
            # thickness grows with the overlap, so the worst one stands out
            pb.radius = 0.08 + min(0.22, 0.18 * c.overlap)
            pb.color = (r, g, b, 255)
            try:
                pb._hitpick_text = ("%.2f over" % c.overlap) if show_overlap else ""
                pb.hitpick_kind = kind
            except Exception:
                pass
        if pbg.num_pseudobonds == 0:
            session.models.close([pbg])
            continue
        if pbg not in session.models:
            session.models.add([pbg])
        made[kind] = pbg.num_pseudobonds
    return made


def label_strain(session, strain, all_atoms):
    """Put a short note on the atoms of each strain finding.

    Eclipsed torsions and puckered rings have no two atoms to draw a line
    between, so they are labelled on the atom instead.
    """
    from chimerax.atomic import Atoms
    from chimerax.core.objects import Objects
    from chimerax.label.label3d import label

    if strain is None or not strain.findings:
        return 0
    _register_attrs(session)
    sel = []
    for f in strain.findings:
        if not f.atoms:
            continue
        anchor = f.atoms[len(f.atoms) // 2]
        if not (0 <= anchor < len(all_atoms)):
            continue
        a = all_atoms[anchor]
        short = {"eclipsed_torsion": "eclipsed", "eclipsed": "eclipsed",
                 "nonplanar_ring": "puckered",
                 "twisted_amide": "twisted amide",
                 "twisted_out_of_conjugation": "out of plane %.0f" % abs(f.value),
                 "forced_planar": "forced flat"}.get(f.kind, f.kind)
        try:
            setattr(a, LABEL_ATTR, short)
        except Exception:
            continue
        sel.append(a)
    if not sel:
        return 0
    try:
        label(session, Objects(atoms=Atoms(sel)), "atoms",
              attribute=LABEL_ATTR, bg_color=_label_backdrop(session),
              color=(200, 90, 30, 255), height=ATOM_LABEL_HEIGHT)
    except Exception:
        return 0
    return len(sel)


def label_interactions(session, interactions=None, all_atoms=None,
                       show_lengths=True, show_angles=False,
                       show_limits=False):
    """Label each interaction line with its own geometry.

    Uses one label call over a per-pseudobond attribute, so distance AND
    angles can differ per line. The previous version labelled whole groups
    with a fixed "{0.length}" format, which is why switching angles on
    changed nothing.
    """
    from chimerax.atomic import PseudobondGroup
    from chimerax.core.objects import Objects
    from chimerax.label.label3d import label

    _register_attrs(session)
    if not show_lengths and not show_angles:
        clear_interaction_labels(session)
        return 0

    n = 0
    for m in list(session.models):
        if not isinstance(m, PseudobondGroup) or not m.name.startswith(GROUP_PREFIX):
            continue
        pbs = m.pseudobonds
        if len(pbs) == 0:
            continue
        for pb in pbs:
            text = getattr(pb, "_hitpick_text", None)
            if text is None:
                try:
                    text = "%.2f A" % pb.length
                except Exception:
                    text = ""
            try:
                setattr(pb, LABEL_ATTR, text)
            except Exception:
                continue
        kind = m.name[len(GROUP_PREFIX):].replace(" ", "_")
        # One label call per band, so the colour can say whether THIS
        # interaction's geometry is within its limits rather than merely
        # repeating what the line colour already says about its type.
        by_band = {}
        for pb in pbs:
            by_band.setdefault(getattr(pb, "_hitpick_band", GOOD), []).append(pb)
        for band, group in by_band.items():
            rgb = BAND_COLORS.get(band, TYPE_COLORS.get(kind, (60, 60, 60)))
            if kind in CLASH_KINDS:
                rgb = TYPE_COLORS[kind]
            from chimerax.atomic import Pseudobonds
            try:
                label(session, Objects(pseudobonds=Pseudobonds(group)),
                      "pseudobonds", attribute=LABEL_ATTR,
                      color=_readable_text_color(session, rgb),
                      bg_color=_label_backdrop(session),
                      height=LABEL_HEIGHT)
                n += len(group)
            except Exception as e:
                session.logger.info("hitpick: could not label %s (%s)"
                                    % (m.name, e))
    return n


# Label size in scene units. Interaction labels cluster in a small volume, so
# they are kept small; atom annotations are smaller still.
LABEL_HEIGHT = 0.35
ATOM_LABEL_HEIGHT = 0.3


def _scene_background(session):
    try:
        bg = session.main_view.background_color
        return tuple(int(255 * c) for c in bg[:3])
    except Exception:
        return (255, 255, 255)


def _label_backdrop(session):
    """A translucent plate in the scene background colour.

    Without it, label text sits directly on sticks and surfaces and is
    effectively unreadable - which is exactly how it looked the first time
    this was rendered.
    """
    r, g, b = _scene_background(session)
    return (r, g, b, 190)


def _readable_text_color(session, rgb):
    """Darken or lighten the type colour until it reads on the backdrop."""
    r, g, b = _scene_background(session)
    light_bg = (0.299 * r + 0.587 * g + 0.114 * b) > 140
    cr, cg, cb = rgb
    if light_bg:
        f = 0.65                      # darken for a light background
        return (int(cr * f), int(cg * f), int(cb * f), 255)
    f = 0.4                           # lighten for a dark one
    return (int(cr + (255 - cr) * f), int(cg + (255 - cg) * f),
            int(cb + (255 - cb) * f), 255)


def clear_interaction_labels(session):
    from chimerax.atomic import PseudobondGroup
    from chimerax.core.objects import Objects
    from chimerax.label.label3d import label_delete
    for m in list(session.models):
        if not isinstance(m, PseudobondGroup) or not m.name.startswith(GROUP_PREFIX):
            continue
        try:
            label_delete(session, Objects(pseudobonds=m.pseudobonds),
                         "pseudobonds")
        except Exception:
            pass


def annotate_atoms(session, labels, all_atoms, clear_first=True):
    """Put handbook values (e.g. pKBHX) on atoms in the 3D view.

    `labels` maps atom index -> text. Only OUR atoms are relabelled; other
    labels in the session are left alone.
    """
    from chimerax.atomic import Atoms
    from chimerax.core.objects import Objects
    from chimerax.label.label3d import label

    _register_attrs(session)
    sel, texts = [], []
    for idx, text in labels.items():
        i = int(idx)
        if not (0 <= i < len(all_atoms)):
            continue
        a = all_atoms[i]
        try:
            setattr(a, LABEL_ATTR, str(text))
        except Exception:
            continue
        sel.append(a)
        texts.append(text)
    if not sel:
        return 0
    atoms = Atoms(sel)
    if clear_first:
        clear_atom_labels(session, all_atoms)
    try:
        label(session, Objects(atoms=atoms), "atoms", attribute=LABEL_ATTR,
              bg_color=_label_backdrop(session), height=ATOM_LABEL_HEIGHT)
    except Exception as e:
        session.logger.warning("hitpick: labelling failed (%s)" % e)
        return 0
    return len(sel)


def clear_atom_labels(session, all_atoms=None):
    """Delete only the atom labels we created."""
    from chimerax.atomic import Atoms
    from chimerax.core.objects import Objects
    from chimerax.label.label3d import label_delete
    if not all_atoms:
        return
    ours = [a for a in all_atoms if getattr(a, LABEL_ATTR, None)]
    if not ours:
        return
    try:
        label_delete(session, Objects(atoms=Atoms(ours)), "atoms")
    except Exception:
        pass


def highlight(session, indices, all_atoms):
    """Select the atoms of a moiety so the user can see what was clicked.

    Flags the change as ours: the handbook also listens for selection changes
    to decide what the user clicked, and highlighting a moiety must not look
    like a new click and recurse.
    """
    from chimerax.atomic import Atoms
    try:
        sel = [all_atoms[int(i)] for i in indices
               if 0 <= int(i) < len(all_atoms)]
    except Exception:
        return 0
    if not sel:
        return 0
    from .state import get_state
    st = get_state(session)
    st._suppress_selection = True
    try:
        session.selection.clear()
        Atoms(sel).selected = True
    finally:
        st._suppress_selection = False
    return len(sel)


def legend_text(counts):
    if not counts:
        return "no interactions detected"
    return "  ".join("%s %d" % (k.replace("_", " "), v)
                     for k, v in sorted(counts.items()))

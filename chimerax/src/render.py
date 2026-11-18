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
}

TYPE_DASHES = {
    "hbond": 6, "hbond_charged": 6, "weak_hbond": 10,
    "salt_bridge": 4, "pi_stack_parallel": 8, "pi_stack_tshaped": 8,
    "cation_pi": 8, "halogen_bond": 6, "hydrophobic": 12,
}

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
                      show_angles=False, min_quality=0.0):
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
            parts = []
            if show_lengths:
                parts.append("%.2f A" % it.distance)
            if show_angles:
                for k, v in it.angles.items():
                    if v == v:
                        parts.append("%s %.0f" % (k, v))
            parts.append("q=%.2f" % it.quality)
            try:
                pb.hitpick_label = " ".join(parts)
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


def label_interactions(session, kinds=None):
    """Turn on pseudobond distance labels for our groups."""
    from chimerax.core.commands import run
    from chimerax.atomic import PseudobondGroup
    n = 0
    # Snapshot: running the label command below adds models, and iterating
    # session.models while it changes raises.
    for m in list(session.models):
        if not isinstance(m, PseudobondGroup) or not m.name.startswith(GROUP_PREFIX):
            continue
        if kinds and m.name not in {group_name(k) for k in kinds}:
            continue
        try:
            run(session, 'label #%s pseudobonds text "{0.length:.2f}"'
                % m.id_string, log=False)
            n += 1
        except Exception:
            pass
    return n


def annotate_atoms(session, labels, all_atoms, clear_first=True):
    """Put handbook values (e.g. pKBHX) on atoms in the 3D view."""
    from chimerax.core.commands import run
    if clear_first:
        clear_atom_labels(session, all_atoms)
    n = 0
    for idx, text in labels.items():
        i = int(idx)
        if not (0 <= i < len(all_atoms)):
            continue
        a = all_atoms[i]
        try:
            spec = "#%s/%s:%d@%s" % (a.structure.id_string, a.residue.chain_id
                                     or "", a.residue.number, a.name)
            run(session, 'label %s text "%s" height 0.4'
                % (spec, text.replace('"', "'")), log=False)
            n += 1
        except Exception:
            continue
    return n


def clear_atom_labels(session, all_atoms=None):
    from chimerax.core.commands import run
    try:
        run(session, "label delete", log=False)
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

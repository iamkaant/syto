"""The `hitpick` commands.

    hitpick setup [receptor #1] [ligand #2]   prepare the binding site
    hitpick analyze [#2]                      analyse one pose and draw it
    hitpick show <types>                      choose which interactions to draw
    hitpick hide <types>
    hitpick lengths true|false
    hitpick labels pKBHX | pKBH+ | off        put handbook values on atoms
    hitpick scale [pKBHX|pKBH+]               where the clicked atom sits
    hitpick explain <atom-spec>               handbook entry for an atom
    hitpick clear
    hitpick report                            text summary of the current pose
"""

from __future__ import annotations

from chimerax.core.commands import (CmdDesc, register, BoolArg, StringArg,
                                    FloatArg, IntArg, ModelArg, ModelsArg,
                                    EmptyArg, Or, RestOfLine)
from chimerax.atomic import AtomsArg, StructureArg
from chimerax.core.errors import UserError

from hitpicking.analysis import ReceptorContext
from hitpicking.nci import criteria as C

from . import adapter, render
from .state import get_state

ALL_TYPES = tuple(C.ALL.keys())


def ensure_ui(session):
    """Register the mouse mode, the selection listener and the ViewDock hook.

    Called from the commands and the panel - never at startup. Touching
    session.ui during bundle initialisation constructs the mouse-mode manager
    before ChimeraX does, which duplicates the 'mouse hover' trigger and takes
    the Toolbar down with it.

    Each piece is checked and installed independently, and re-checked on every
    call. A single "already done" flag meant that if any one piece failed, or
    the state object was later replaced, nothing was ever retried and the
    feature silently stayed missing with no way back.
    """
    ui = getattr(session, "ui", None)
    if ui is None or not getattr(ui, "is_gui", False):
        return                      # nogui: nothing to register
    if not hasattr(ui, "mouse_modes"):
        return                      # UI not built yet; try again next time

    st = get_state(session)

    try:
        from .mousemode import register_mouse_mode
        register_mouse_mode(session)
    except Exception as e:
        session.logger.info("hitpick: mouse mode unavailable (%s)" % e)

    if getattr(st, "_sel_handler", None) is None:
        try:
            _install_selection_listener(session)
        except Exception as e:
            session.logger.info(
                "hitpick: selection listener unavailable (%s)" % e)

    try:
        from .hover import install as install_hover, is_installed
        if st.options.hover_detail and not is_installed(session):
            install_hover(session)
    except Exception as e:
        session.logger.info("hitpick: hover detail unavailable (%s)" % e)

    try:
        from .viewdock_link import autohook, is_hooked
        if not is_hooked(session):
            autohook(session)
    except Exception as e:
        session.logger.info("hitpick: ViewDock link not established (%s)" % e)


def _pose_summary(result):
    if result is None:
        return None
    out = {
        "name": result.name,
        "counts": dict(result.counts),
        "n_interactions": len(result.interactions),
        "warnings": list(result.warnings),
    }
    if result.strain is not None:
        out["strain"] = result.strain.counts
        out["worst_overlap"] = round(result.strain.worst_overlap, 2)
    return out


def _state_summary(st):
    return {
        "ready": st.ready,
        "receptor": getattr(st.receptor_model, "name", None),
        "ligand": getattr(st.ligand_model, "name", None),
        "enabled": sorted(st.options.enabled_nci),
        "handbook": st.handbook.describe() if st.handbook else None,
        "pose": _pose_summary(st.result),
    }


def _install_selection_listener(session):
    """Explain an atom whenever the user selects exactly one.

    Registering a mouse mode is not enough: a mode does nothing until it is
    bound to a button, and silently rebinding the user's mouse would be
    rude. Selecting an atom (ctrl-click, by default) already works out of the
    box, so that is what drives the handbook.
    """
    from chimerax.core.selection import SELECTION_CHANGED
    st = get_state(session)
    if getattr(st, "_sel_handler", None) is not None:
        return

    def on_selection(trigger_name, data):
        s = get_state(session)
        if not s.options.handbook_on_click or s.molsys is None:
            return
        if getattr(s, "_suppress_selection", False):
            return          # this selection is our own highlight, not a click
        if s.handbook is None:
            return
        # A bond selection asks a different question from an atom one: not
        # "what is this group" but "what is this torsion", so it drives the
        # torsion profile instead of the handbook entry.
        try:
            bonds = session.selection.items("bonds")
        except Exception:
            bonds = None
        if bonds and len(bonds[0]) == 1:
            if _explain_bond(session, s, bonds[0][0]):
                return

        try:
            sel = session.selection.items("atoms")
        except Exception:
            return
        if not sel:
            return
        atoms = sel[0]
        if len(atoms) != 1:
            return           # a whole residue or more: not a moiety question
        idx = s.atom_index(atoms[0])
        if idx is None:
            return
        try:
            hit = s.handbook.for_atom(s.molsys, idx)
        except Exception as e:
            session.logger.info("hitpick: %s" % e)
            return
        s.last_hit = hit
        s.last_torsion = None      # an atom click is not a torsion question
        _log_hit(session, hit)
        s.notify()

    st._sel_handler = session.triggers.add_handler(SELECTION_CHANGED,
                                                   on_selection)


def _explain_bond(session, st, bond):
    """Show the torsion measured across a clicked bond, if there is one."""
    if st.result is None or st.result.strain is None or st.molsys is None:
        return False
    try:
        a1, a2 = bond.atoms
    except Exception:
        return False
    i1, i2 = st.atom_index(a1), st.atom_index(a2)
    if i1 is None or i2 is None:
        return False
    key = tuple(sorted((i1, i2)))
    for t in st.result.strain.torsions:
        if tuple(sorted(t.bond)) != key:
            continue
        st.last_torsion = t
        render.show_torsion(session, t, st.all_atoms)
        ms = st.molsys
        session.logger.info(
            "<b>%s &mdash; %s - %s</b>"
            % (t.subtype or t.bond_type, ms.atom_label(t.bond[0]),
               ms.atom_label(t.bond[1])), is_html=True)
        session.logger.info(
            "   measured %.1f deg over %s"
            % (t.angle, " - ".join(ms.atom_label(a) for a in t.atoms)))
        if t.note:
            session.logger.info("   " + t.note)
        from hitpicking.strain import torsions as T
        view = T.profile_view(t.bond_type, t.subtype, t.ortho_substituted,
                              t.angle)
        if view is not None:
            for line in T.profile_text(view):
                session.logger.info(line.replace(" ", "&nbsp;"), is_html=True)
        st.notify()
        return True
    return False


def hitpick_autoexplain(session, enable=True):
    """Turn the explain-on-select behaviour on or off."""
    ensure_ui(session)
    st = get_state(session)
    st.options.handbook_on_click = bool(enable)
    session.logger.info("hitpick: explain-on-select is %s"
                        % ("on" if enable else "off"))
    return st.options.handbook_on_click


hitpick_autoexplain_desc = CmdDesc(
    optional=[("enable", BoolArg)],
    synopsis="Explain an atom automatically when it is selected")


# --------------------------------------------------------------------- setup

def hitpick_setup(session, receptor=None, ligand=None):
    ensure_ui(session)
    st = get_state(session)

    rec = receptor
    if rec is None:
        rec = adapter.find_receptor_structure(session)
    if rec is None:
        raise UserError(
            "No receptor found. Open one, or name it: hitpick setup #1")

    st.receptor_model = rec
    ms, atoms = adapter.from_structures([rec])
    st.context = ReceptorContext(ms, st.options)
    st.context._cx_atoms = atoms

    log = session.logger
    log.info("hitpick: receptor %s, %d atoms" % (rec.name, ms.n_atoms))
    rep = ms.meta.get("repaired_disulfides")
    if rep:
        log.warning("hitpick: added %d disulfide bond(s) the file did not "
                    "declare: %s" % (len(rep), rep))
    for w in st.context.warnings:
        log.warning("hitpick: " + w)
    if st.handbook is None:
        log.warning("hitpick: handbook data not built; run "
                    "tools/build_data.py in the source tree")
    else:
        log.info("hitpick: " + st.handbook.describe())

    log.info("hitpick: select a single atom (ctrl-click) to see its handbook "
             "entry. For a dedicated button: "
             '<b>mousemode right "hit picking handbook"</b>', is_html=True)

    if ligand is not None:
        hitpick_analyze(session, ligand)
    return _state_summary(st)


hitpick_setup_desc = CmdDesc(
    optional=[("receptor", StructureArg), ("ligand", StructureArg)],
    synopsis="Prepare a binding site for hit picking")


# ------------------------------------------------------------------- analyze

def hitpick_analyze(session, ligand=None):
    """Analyse one docked pose and draw its interactions.

    Reentrancy guard: analysing adds pseudobond groups and updates the
    ViewDock table, either of which can make ViewDock re-emit
    selection_changed, which calls straight back into here. Without the guard
    that recurses until Python gives up, and the failure is ugly and remote -
    it took out the REST bridge rather than reporting itself.
    """
    if getattr(session, "_hitpick_analyzing", False):
        return _pose_summary(get_state(session).result)
    ensure_ui(session)
    st = get_state(session)
    if not st.ready:
        hitpick_setup(session)
        st = get_state(session)

    lig = ligand
    if lig is None:
        lig = _guess_displayed_ligand(session, st)
    if lig is None:
        raise UserError("No ligand found. Name one: hitpick analyze #2")

    session._hitpick_analyzing = True
    try:
        return _analyze_pose(session, st, lig)
    finally:
        session._hitpick_analyzing = False


def _analyze_pose(session, st, lig):

    st.ligand_model = lig
    # Receptor + this pose, in one index space: receptor first, as the core
    # expects, so the cached receptor perception stays valid.
    ms, atoms = adapter.from_structures(
        [st.receptor_model, lig], ligand_structures=[lig])
    st.molsys = ms
    st.all_atoms = atoms

    from hitpicking.nci.detect import detect_interactions
    from hitpicking.perceive.features import perceive_all
    feats = perceive_all(ms)
    res = detect_interactions(ms, enabled=st.options.enabled_nci,
                              include_weak_hbonds=st.options.include_weak_hbonds,
                              features=feats)

    strain = None
    if st.options.detect_strain:
        from hitpicking.strain import analyze_strain
        strain = analyze_strain(ms)

    from hitpicking.analysis import PoseResult
    n_rec = st.receptor_model.num_atoms
    st.result = PoseResult(lig.name, res.interactions, res.warnings,
                           res.counts, adapter.pose_scores(lig),
                           combined=ms, n_receptor=n_rec, strain=strain)

    _redraw(session, st)
    session.logger.info("hitpick: %s -> %s"
                        % (lig.name, st.result.summary()))
    for w in res.warnings:
        session.logger.warning("hitpick: " + w)
    st.notify()
    return _pose_summary(st.result)


def _guess_displayed_ligand(session, st):
    cands = adapter.find_ligand_structures(
        session, exclude=[st.receptor_model] if st.receptor_model else [])
    shown = [m for m in cands if m.display]
    if len(shown) == 1:
        return shown[0]
    if len(cands) == 1:
        return cands[0]
    return shown[0] if shown else None


def _analyze_desc():
    # A CmdDesc binds to exactly one command, so the British-spelling alias
    # needs its own instance rather than a second registration of this one.
    return CmdDesc(optional=[("ligand", StructureArg)],
                   synopsis="Analyse one docked pose and draw its interactions")


hitpick_analyze_desc = _analyze_desc()


# --------------------------------------------------------------------- draw

def _redraw(session, st):
    if st.result is None:
        return
    render.set_line_width(session, st.options.line_width)
    render.draw_interactions(
        session, st.result.interactions, st.all_atoms,
        show_lengths=st.options.show_lengths,
        show_angles=st.options.show_angles,
        show_limits=st.options.show_limits)
    render.label_interactions(session,
                              show_lengths=st.options.show_lengths,
                              show_angles=st.options.show_angles,
                              show_limits=st.options.show_limits)
    if st.options.show_clashes:
        drawn = render.draw_clashes(session, st.result.strain, st.all_atoms,
                                    show_overlap=st.options.show_lengths)
        if drawn and st.options.show_lengths:
            render.label_interactions(
                session, show_lengths=st.options.show_lengths,
                show_angles=st.options.show_angles,
                show_limits=st.options.show_limits)
    else:
        render.clear_groups(session, render.CLASH_KINDS)
    if st.options.show_strain_labels and st.options.annotate_atom_values is False:
        render.label_strain(session, st.result.strain, st.all_atoms,
                            unsatisfied=st.result.unsatisfied)


def _parse_types(text):
    if not text:
        return set()
    out = set()
    for tok in text.replace(",", " ").split():
        t = tok.strip().lower()
        if t in ("all", "*"):
            return set(ALL_TYPES)
        if t in ("default", "defaults"):
            return set(C.DEFAULT_ENABLED)
        if t in ALL_TYPES:
            out.add(t)
            continue
        matches = [k for k in ALL_TYPES if k.startswith(t)]
        if len(matches) == 1:
            out.add(matches[0])
        elif t in ("stacking", "pi", "pistack"):
            out |= {"pi_stack_parallel", "pi_stack_tshaped"}
        elif t in ("hbonds", "hydrogen"):
            out |= {"hbond", "hbond_charged"}
        else:
            raise UserError("unknown interaction type %r; known: %s"
                            % (tok, ", ".join(ALL_TYPES)))
    return out


def hitpick_show(session, types=None):
    st = get_state(session)
    want = _parse_types(types) if types else set(C.DEFAULT_ENABLED)
    st.options.enabled_nci |= want
    session.logger.info("hitpick: showing %s"
                        % ", ".join(sorted(st.options.enabled_nci)))
    if st.result is not None:
        hitpick_analyze(session, st.ligand_model)
    return sorted(st.options.enabled_nci)


hitpick_show_desc = CmdDesc(optional=[("types", RestOfLine)],
                            synopsis="Enable interaction types")


def hitpick_hide(session, types=None):
    st = get_state(session)
    drop = _parse_types(types) if types else set(ALL_TYPES)
    st.options.enabled_nci -= drop
    session.logger.info("hitpick: showing %s"
                        % (", ".join(sorted(st.options.enabled_nci)) or "nothing"))
    if st.result is not None:
        hitpick_analyze(session, st.ligand_model)
    return sorted(st.options.enabled_nci)


hitpick_hide_desc = CmdDesc(optional=[("types", RestOfLine)],
                            synopsis="Disable interaction types")


def hitpick_lengths(session, show=True):
    st = get_state(session)
    st.options.show_lengths = bool(show)
    if st.result is not None:
        _redraw(session, st)
    return st.options.show_lengths


hitpick_lengths_desc = CmdDesc(optional=[("show", BoolArg)],
                               synopsis="Show or hide interaction lengths")


def hitpick_reveal(session, show=True):
    """Display the residues our interactions point at, or put them back.

    `show false` hides exactly the atoms this command turned on, and
    nothing the user displayed themselves. Before, it took the argument
    and ignored it: there was no way back.
    """
    ensure_ui(session)
    st = get_state(session)
    if st.result is None:
        raise UserError("nothing analysed yet; run: hitpick analyze")
    n = render.reveal_partners(session, st.result.interactions, st.all_atoms,
                               show=bool(show))
    st.options.reveal_partners = bool(show)
    session.logger.info("hitpick: %s %d atom(s)"
                        % ("revealed" if show else "hid again", n))
    st.notify()
    return n


hitpick_reveal_desc = CmdDesc(
    optional=[("show", BoolArg)],
    synopsis="Display the residues the interactions point at")


def hitpick_width(session, width=None):
    """Set the thickness of the interaction and clash lines."""
    ensure_ui(session)
    st = get_state(session)
    if width is None:
        session.logger.info(
            "hitpick: line width is %.3f (ChimeraX draws its own hydrogen "
            "bonds at %.3f). Set it with: hitpick width 0.05"
            % (st.options.line_width, render.DEFAULT_LINE_WIDTH))
        return st.options.line_width
    st.options.line_width = render.set_line_width(session, width)
    if st.result is not None:
        _redraw(session, st)
    session.logger.info("hitpick: line width %.3f" % st.options.line_width)
    return st.options.line_width


hitpick_width_desc = CmdDesc(
    optional=[("width", FloatArg)],
    synopsis="Set the thickness of interaction and clash lines")


def hitpick_panel(session, floating=True):
    """Detach the handbook panel, so it can be widened to two columns."""
    from .tool import HandbookTool
    tools = [t for t in session.tools.list() if isinstance(t, HandbookTool)]
    if not tools:
        from chimerax.core.commands import run
        run(session, 'ui tool show "Hit Picking Handbook"')
        tools = [t for t in session.tools.list() if isinstance(t, HandbookTool)]
    if not tools:
        raise UserError("could not open the Hit Picking Handbook panel")
    ok = tools[0].float_window(floating)
    if ok:
        session.logger.info(
            "hitpick: panel %s. Widen it past ~680 px and it lays out in two "
            "columns; past ~1100 px, three."
            % ("detached" if floating else "docked"))
    return ok


hitpick_panel_desc = CmdDesc(
    optional=[("floating", BoolArg)],
    synopsis="Detach or re-dock the handbook panel")


def hitpick_torsionprofile(session, all=False):
    """Plot the QM energy profile for the torsions of this pose."""
    ensure_ui(session)
    st = get_state(session)
    if st.result is None or st.result.strain is None:
        raise UserError("nothing analysed yet; run: hitpick analyze")
    from hitpicking.strain import torsions as T
    if T.qm_reference() is None:
        raise UserError(
            "no QM torsion reference built. Run tools/fetch_qm_torsions.py "
            "then tools/build_torsion_profiles.py")

    ts = st.result.strain.torsions
    shown = 0
    seen = set()
    ms = st.molsys
    for t in sorted(ts, key=lambda x: (not x.flagged, x.subtype or "")):
        if not (all or t.flagged):
            continue
        key = (t.bond_type, t.subtype, t.ortho_substituted)
        if key in seen and not all:
            continue
        seen.add(key)
        view = T.profile_view(t.bond_type, t.subtype, t.ortho_substituted,
                              t.angle)
        if view is None:
            continue
        session.logger.info(
            "<b>%s &mdash; %s</b>"
            % (view["label"], "%s - %s" % (ms.atom_label(t.bond[0]),
                                           ms.atom_label(t.bond[1]))),
            is_html=True)
        for line in T.profile_text(view):
            session.logger.info(line.replace(" ", "&nbsp;"), is_html=True)
        shown += 1
    if not shown:
        session.logger.info(
            "hitpick: no torsion in this pose has a QM profile"
            + ("" if all else " among the flagged ones; try "
               "'hitpick torsionprofile true'"))
    else:
        qm = T.qm_reference()
        src = (qm.get("attribution", {}) or {})
        session.logger.info(
            "relaxed QM torsion drives from %s, %s"
            % (src.get("source", "QCArchive"),
               src.get("dataset_licence", "see attribution")))
        session.logger.info(
            "gas-phase and intrinsic: a protein may well pay a couple of "
            "kcal/mol for a better contact.")
    return {"profiles": shown}


hitpick_torsionprofile_desc = CmdDesc(
    optional=[("all", BoolArg)],
    synopsis="Plot the QM energy profile for this pose's torsions")


def hitpick_hover(session, show=True):
    """Put hitpick detail in ChimeraX's hover balloon, or give it back."""
    ensure_ui(session)
    st = get_state(session)
    st.options.hover_detail = bool(show)
    from . import hover
    if show:
        ok = hover.install(session)
        if not ok and not hover.is_installed(session):
            session.logger.warning(
                "hitpick: the 'mouse hover' trigger does not exist yet - "
                "hover over the 3D view once, then try again.")
            return False
        session.logger.info(
            "hitpick: hovering an atom now shows its role and basicity, and "
            "hovering an interaction shows the measured geometry against its "
            "limits.")
    else:
        hover.remove(session)
        session.logger.info("hitpick: ChimeraX's own hover balloon restored.")
    return st.options.hover_detail


hitpick_hover_desc = CmdDesc(
    optional=[("show", BoolArg)],
    synopsis="Show hitpick detail in the hover balloon")


def hitpick_torsions(session, all=False):
    """List every rotatable torsion, with a judgement where one is possible."""
    ensure_ui(session)
    st = get_state(session)
    if st.result is None or st.result.strain is None:
        raise UserError("nothing analysed yet; run: hitpick analyze")
    ts = st.result.strain.torsions
    if not ts:
        session.logger.info("hitpick: no rotatable torsions in this ligand")
        return {"n": 0}

    ms = st.molsys
    log = session.logger
    log.info("<b>torsions &mdash; %s</b>" % st.result.name, is_html=True)
    shown = 0
    for t in sorted(ts, key=lambda x: (not x.flagged, x.bond_type)):
        if not (all or t.flagged):
            continue
        shown += 1
        # Two tiers, not one. A torsion the QM scan for its own bond class
        # measures past QM_WARN_KCAL is costing the pose real energy; one
        # between the two thresholds is worth saying and not worth alarm.
        mark = {"warning": "!!", "note": " !"}.get(t.severity, "  ")
        cost = ("  %+.1f kcal/mol" % t.energy) if t.energy is not None else ""
        log.info("  %s %-16s %7.1f deg%s   %s - %s"
                 % (mark, t.subtype or t.bond_type, t.angle, cost,
                    ms.atom_label(t.bond[0]), ms.atom_label(t.bond[1])))
        if t.note:
            log.info("      %s" % t.note)
    if not all:
        from hitpicking.strain.torsions import QM_WARN_KCAL
        n_un = sum(1 for t in ts if t.assessment == "unjudged")
        n_warn = sum(1 for t in ts if t.severity == "warning"
                     and t.energy is not None)
        log.info("  %d torsion(s) total; %d flagged%s. Use 'hitpick "
                 "torsions true' for all of them."
                 % (len(ts), shown,
                    (", %d (!!) measured above %.0f kcal/mol"
                     % (n_warn, QM_WARN_KCAL)) if n_warn else ""))
        if n_un:
            log.info("  %d could not be judged: there is no redistributable "
                     "CSD torsion library, so bond types without an "
                     "unambiguous preference are reported, not scored."
                     % n_un)
    return {"n": len(ts), "shown": shown}


hitpick_torsions_desc = CmdDesc(
    optional=[("all", BoolArg)],
    synopsis="List rotatable torsions and flag the strained ones")


def hitpick_limits(session, show=True):
    """Put the literature limits in the label text, not just in its colour."""
    ensure_ui(session)
    st = get_state(session)
    st.options.show_limits = bool(show)
    if st.result is not None:
        _redraw(session, st)
    session.logger.info(
        "hitpick: limits are %s in the labels. Label colour always shows "
        "how the geometry sits in its envelope: green comfortably inside, "
        "amber well off ideal, red only just inside."
        % ("shown" if show else "hidden"))
    return st.options.show_limits


hitpick_limits_desc = CmdDesc(
    optional=[("show", BoolArg)],
    synopsis="Show the literature limits alongside each measured value")


def hitpick_scaleplot(session, show=True):
    """Show or hide the basicity scale plot in the panel."""
    ensure_ui(session)
    st = get_state(session)
    st.options.show_scale = bool(show)
    st.notify()
    session.logger.info("hitpick: basicity scale plot is %s"
                        % ("shown" if show else "hidden"))
    return st.options.show_scale


hitpick_scaleplot_desc = CmdDesc(
    optional=[("show", BoolArg)],
    synopsis="Show or hide the basicity scale plot in the panel")


def hitpick_clashes(session, show=True):
    """Draw or hide clashes in the 3D view."""
    ensure_ui(session)
    st = get_state(session)
    st.options.show_clashes = bool(show)
    st.options.detect_strain = bool(show) or st.options.detect_strain
    if st.result is not None:
        hitpick_analyze(session, st.ligand_model)
    else:
        if not show:
            render.clear_groups(session, render.CLASH_KINDS)
    session.logger.info("hitpick: clashes are %s"
                        % ("shown" if show else "hidden"))
    return st.options.show_clashes


hitpick_clashes_desc = CmdDesc(
    optional=[("show", BoolArg)],
    synopsis="Show or hide van der Waals clashes")


def hitpick_angles(session, show=True):
    st = get_state(session)
    st.options.show_angles = bool(show)
    if st.result is not None:
        _redraw(session, st)
    return st.options.show_angles


hitpick_angles_desc = CmdDesc(optional=[("show", BoolArg)],
                              synopsis="Show or hide interaction angles")


# -------------------------------------------------------------- annotations

def hitpick_labels(session, quantity="pKBHX"):
    """Put handbook values on the ligand atoms in 3D."""
    st = get_state(session)
    if st.handbook is None:
        raise UserError("handbook data not built; run tools/build_data.py")
    if st.molsys is None:
        raise UserError("nothing analysed yet; run: hitpick analyze")
    q = (quantity or "").strip()
    if q.lower() in ("off", "none", "clear"):
        render.clear_atom_labels(session, st.all_atoms)
        st.options.annotate_atom_values = False
        return 0
    labels = st.handbook.annotate_system(st.molsys, q, ligand_only=True)
    if not labels:
        session.logger.warning(
            "hitpick: no %s values apply to this ligand's atoms" % q)
        return 0
    n = render.annotate_atoms(session, labels, st.all_atoms)
    st.options.annotate_atom_values = True
    session.logger.info("hitpick: labelled %d atoms with %s" % (n, q))
    session.logger.info(
        "hitpick: each label shows the UNSUBSTITUTED PARENT compound's value "
        "where the data has it, with the family range in brackets. It is not "
        "a value for this exact compound: perception identifies the ring or "
        "group, not the substitution pattern, and substituents move %s by "
        "a log unit or more. A * marks values flagged for review." % q)
    if q == "pKBH+":
        session.logger.info(
            "hitpick: pKBH+ is the pKa of the conjugate acid, not a "
            "hydrogen-bond strength - a group above about 7.4 is mostly "
            "protonated at physiological pH, so it donates and can make a "
            "salt bridge rather than accepting. The two scales barely "
            "correlate; for acceptor strength label pKBHX instead.")
    return n


hitpick_labels_desc = CmdDesc(optional=[("quantity", StringArg)],
                              synopsis="Label atoms with handbook values")


# ------------------------------------------------------------------ explain

def hitpick_explain(session, atoms=None):
    st = get_state(session)
    if st.handbook is None:
        raise UserError("handbook data not built; run tools/build_data.py")
    if st.molsys is None:
        raise UserError("nothing analysed yet; run: hitpick analyze")

    if atoms is None or len(atoms) == 0:
        sel = session.selection.items("atoms")
        atoms = sel[0] if sel else None
    if atoms is None or len(atoms) == 0:
        raise UserError("select an atom, or: hitpick explain /A:100@OD1")

    a = atoms[0]
    idx = st.atom_index(a)
    if idx is None:
        raise UserError("that atom is not part of the analysed system; "
                        "run hitpick analyze first")
    hit = st.handbook.for_atom(st.molsys, idx)
    st.last_hit = hit
    _log_hit(session, hit)
    if hit.highlight:
        render.highlight(session, hit.highlight, st.all_atoms)
    st.notify()
    return hit.to_dict()


def _log_hit(session, hit):
    log = session.logger
    log.info("<b>%s</b>" % hit.title, is_html=True)
    if not hit.entries:
        for w in hit.warnings:
            log.info("   " + w)
        return
    for a in hit.annotations:
        flag = "  (%d of %d values flagged for review)" % (
            a["n_flagged"], a["n"]) if a["n_flagged"] else ""
        log.info("   %s: median %.2f, range %.2f to %.2f over %d compounds%s"
                 % (a["quantity"], a["median"], a["min"], a["max"],
                    a["n"], flag))
    for e in hit.entries[:4]:
        log.info("   <i>%s</i>" % e.get("title", e["id"]), is_html=True)
        s = (e.get("summary") or "").strip()
        if s:
            log.info("      " + " ".join(s.split())[:300])
        for n in (e.get("practical_notes") or [])[:2]:
            log.info("      - " + " ".join(str(n).split())[:240])
    for w in hit.warnings:
        log.warning("   " + w)


hitpick_explain_desc = CmdDesc(optional=[("atoms", Or(AtomsArg, EmptyArg))],
                               synopsis="Handbook entry for an atom")


# ------------------------------------------------------------------- report

def hitpick_scale(session, quantity="pKBHX", all_elements=False):
    """Print a scale, with the last clicked moiety marked.

    Two are worth asking for and they are not interchangeable:

      pKBHX   how strongly the atom accepts a hydrogen bond
      pKBH+   whether the atom is protonated at pH 7.4

    Across families these barely correlate, so a strong acceptor is not
    necessarily a strong base, and reading one off the other is the mistake
    this pair of plots exists to prevent.
    """
    ensure_ui(session)
    st = get_state(session)
    if st.handbook is None:
        raise UserError("handbook data not built; run tools/build_data.py")
    mids = list(st.last_hit.moiety_ids) if st.last_hit else []
    lines = st.handbook.scale_text(quantity, mids,
                                   same_element_only=not all_elements)
    if not lines:
        from hitpicking.handbook.scale import QUANTITY_INFO
        session.logger.warning(
            "hitpick: no %s data to plot (the scales with data are: %s)"
            % (quantity, ", ".join(sorted(QUANTITY_INFO))))
        return None
    if mids:
        session.logger.info("<b>%s scale &mdash; %s</b>"
                            % (quantity, st.last_hit.title), is_html=True)
    else:
        session.logger.info("<b>%s scale (click an atom to mark one)</b>"
                            % quantity, is_html=True)
    for ln in lines:
        session.logger.info(ln.replace(" ", "&nbsp;"), is_html=True)
    session.logger.info("values from Laurence et al. 2009, "
                        "<a href='https://doi.org/10.1021/jm801331y'>"
                        "10.1021/jm801331y</a>", is_html=True)
    return {"quantity": quantity, "rows": len(lines)}


hitpick_scale_desc = CmdDesc(
    optional=[("quantity", StringArg), ("all_elements", BoolArg)],
    synopsis="Show where the clicked atom sits on the basicity scale")


def hitpick_report(session):
    st = get_state(session)
    if st.result is None:
        raise UserError("nothing analysed yet; run: hitpick analyze")
    r = st.result
    log = session.logger
    log.info("<b>hitpick: %s</b>" % r.name, is_html=True)
    log.info("   " + render.legend_text(r.counts))
    for it in r.interactions:
        bits = ["%-18s" % it.kind, "q=%.2f" % it.quality,
                "d=%.2f A" % it.distance]
        for k, v in it.angles.items():
            if v == v:
                bits.append("%s=%.0f" % (k, v))
        log.info("   " + "  ".join(bits) + "   " + it.label)
        for f in it.flags:
            log.info("        ! " + f)
    if r.strain is not None:
        sr = r.strain
        if sr.clashes or sr.internal or sr.findings:
            log.info("   <b>clashes and strain</b>", is_html=True)
        for c in sr.clashes + sr.internal:
            log.info("   %-15s d=%.2f A  overlap %.2f A%s   %s"
                     % (c.kind, c.distance, c.overlap,
                        "  SEVERE" if c.severe else "", c.label))
        for f in sr.findings:
            log.info("   %-15s %s" % (f.severity, f.message))
    if r.unsatisfied:
        log.info("   <b>buried polar atoms with no H-bond partner</b>",
                 is_html=True)
        for u in r.unsatisfied:
            log.info("   %-15s %s" % (u.side, u.message))
    for w in r.warnings:
        log.warning("   " + w)
    return _pose_summary(r)


hitpick_report_desc = CmdDesc(synopsis="Text report for the current pose")


# ---------------------------------------------------------------- learning
#
# Every one of these imports .learn lazily. A user who only wants the
# handbook never loads the model, the feature code or numpy's linear algebra.

def hitpick_learn_panel(session):
    """Open the active-learning panel."""
    from chimerax.core.commands import run
    run(session, 'ui tool show "Hit Picking Learn"')
    return True


hitpick_learn_panel_desc = CmdDesc(
    synopsis="Open the active-learning panel")


def _job_feedback(session, title):
    """Progress to the status line and the Learn panel, if it is open."""
    def on_progress(done, total, message):
        session.logger.status("hitpick %s: %s" % (title, message))
        _refresh_learn_panel(session)
    return on_progress


def _refresh_learn_panel(session):
    import sys
    # Only if the panel is actually open: importing it here would drag the
    # whole AL stack in for someone who never asked for it.
    mod = sys.modules.get("chimerax.syto.learn_tool")
    if mod is None:
        return
    try:
        tools = session.tools.find_by_class(mod.LearnTool)
    except Exception:
        return
    for t in tools:
        try:
            t.refresh()
        except Exception:
            pass


def hitpick_learn_collect(session, limit=None, wait=False):
    """Analyse and featurise the poses, ready for learning.

    At about 0.14 s a pose this is minutes of work on a full party, so in a
    window it runs on a background thread and the command returns at once.
    `wait true` forces the old blocking behaviour for scripts.
    """
    ensure_ui(session)
    from . import learn, background

    job = background.claim(session, "analysing")

    def work(j):
        return learn.collect(session, limit=limit, quiet=True, job=j)

    def finished(j):
        background.release(session, j)
        if j.error is not None:
            session.logger.warning("hitpick learn: analysing failed: %s"
                                   % j.error)
        elif j.cancelled:
            d = learn.describe(session)
            session.logger.info(
                "hitpick learn: cancelled, %d pose(s) featurised so far"
                % d["poses"])
        else:
            d = learn.describe(session)
            session.logger.info(
                "hitpick learn: %d pose(s) featurised, %d in total"
                % (j.result or 0, d["poses"]))
            if d["poses"] and not d["can_train"]:
                session.logger.info("   %s" % d["why_not"])
            elif d["can_train"]:
                session.logger.info("   ready: hitpick learn train")
        _refresh_learn_panel(session)

    threaded = job.start(work, on_progress=_job_feedback(session, "learn"),
                         on_done=finished, threaded=(False if wait else None))
    if threaded:
        session.logger.info(
            "hitpick learn: analysing in the background. The window stays "
            "usable; 'hitpick learn cancel' stops it.")
        return {"started": True, "background": True}
    return learn.describe(session)


hitpick_learn_collect_desc = CmdDesc(
    optional=[("limit", IntArg)],
    keyword=[("wait", BoolArg)],
    synopsis="Analyse and featurise poses for active learning")


def hitpick_learn_apply(session):
    """Write the model's predictions into ViewDock's table."""
    ensure_ui(session)
    from . import learn
    n, why = learn.apply_predictions(session)
    if n:
        session.logger.info(
            "hitpick learn: scored %d pose(s) in ViewDock. Sort by 'HP Pred "
            "Rank' (2 = good) or 'HP Uncert' to see what it is unsure of."
            % n)
    else:
        session.logger.warning("hitpick learn: %s" % (why or "nothing to do"))
    return {"written": n, "why": why}


hitpick_learn_apply_desc = CmdDesc(
    synopsis="Write the model's predictions into the ViewDock table")


def hitpick_learn_cancel(session):
    """Stop whatever background analysis or training is running."""
    from . import background
    job = background.current(session)
    if job is None:
        session.logger.info("hitpick learn: nothing is running")
        return {"cancelled": False}
    session.logger.info("hitpick learn: cancelling %s - it will stop after "
                        "the pose in hand" % job.title)
    background.cancel(session)
    return {"cancelled": True, "was": job.title}


hitpick_learn_cancel_desc = CmdDesc(
    synopsis="Stop background analysis or training")


def hitpick_learn_grade(session, grade, structure=None):
    """Grade the displayed pose without ViewDock."""
    ensure_ui(session)
    from . import learn
    st = get_state(session)
    ls = learn.state(session)
    s = structure or st.ligand_model
    if s is None:
        raise UserError("no pose to grade; run hitpick analyze first")
    key = ls.key_for(s)
    if key not in ls.learner.features:
        if st.result is None:
            raise UserError("analyse the pose first: hitpick analyze")
        ls.add(s, st.result)
    try:
        ls.learner.grade(key, grade)
    except ValueError as e:
        raise UserError(str(e))
    from . import viewdock_link as vl
    rank = ls.learner.grades.rank(key)
    vl.set_rating(session, s, rank + 1)
    session.logger.info("hitpick learn: %s graded %s" % (s.name, grade))
    return ls.learner.grades.counts()


def hitpick_learn_assume(session, grade="good", forget=False,
                         check=True):
    """Grade every unrated pose, for a list that was walked end to end.

    Rating by exception is a normal way to work a hit-picking list: press b
    or m on what you object to, leave the rest. The model never sees that.
    Trained on objections alone it learns a good account of what you reject
    and nothing at all about what you would buy, and "HP Pred Rank" has no
    2 to offer because nothing ever showed it one.

    This says the quiet part out loud. Only use it on a list you went
    through to the end: on a half-finished one it labels everything you
    have not looked at yet, which is worse than no labels. That is not a
    figure of speech - on a session where the premise did not hold, taking
    it up moved held-out bad-vs-maybe AUC from 0.82 to 0.40.

    So when there is a trained model to ask, the premise is tested before
    it is taken up: if the unrated poses were really your good ones, the
    model should already rank them above every pose you graded. `check
    false` skips that.

    `forget true` takes it back and leaves the clicked grades alone.
    """
    ensure_ui(session)
    from . import learn
    ls = learn.state(session)
    log = session.logger

    if forget:
        n = ls.forget_assumed()
        log.info("hitpick learn: dropped %d assumed grade(s); %d grade(s) "
                 "given by hand remain" % (n, len(ls.learner.grades)))
        _refresh_learn_panel(session)
        return {"forgotten": n}

    if not len(ls.learner.features):
        raise UserError("no poses analysed yet; run: hitpick learn collect")

    evidence = None
    if check:
        evidence = learn.assumption_evidence(session)
        _report_assumption_evidence(log, evidence, grade)

    try:
        n = ls.assume_unrated(grade)
    except ValueError as e:
        raise UserError(str(e))

    human, assumed = ls.human_counts()
    n_human = sum(human.values())
    log.info("hitpick learn: assumed <b>%s</b> for %d unrated pose(s)"
             % (grade, n), is_html=True)
    log.info("hitpick learn: by hand %s; assumed %s"
             % (_grade_line(human), _grade_line(assumed)))
    if n_human and n > 3 * n_human:
        log.warning(
            "hitpick learn: %d assumed label(s) against %d given by hand. "
            "The assumption now IS the training set, so if it is wrong the "
            "model will be too - on a session where it was wrong, held-out "
            "bad-vs-maybe AUC fell from 0.82 to 0.40, which is worse than "
            "guessing." % (n, n_human))
    log.warning("hitpick learn: this is an assertion about YOUR list, not a "
                "measurement. It is right only if you went through every "
                "pose. Undo it with: hitpick learn assume forget true")
    _refresh_learn_panel(session)
    return {"assumed": n, "grade": grade, "evidence": evidence}


def _report_assumption_evidence(log, ev, grade):
    """Say whether the trained model agrees with what is being asserted."""
    if not ev:
        log.info("hitpick learn: no trained model to check the assumption "
                 "against. Train first and run this again to have it tested.")
        return
    log.info("hitpick learn: checking the assumption against the model "
             "trained on the %d grade(s) you gave..." % ev["n_human"])
    worrying = []
    for label, v in ev["versus"].items():
        verdict = ("above" if v["auc"] > 0.55 else
                   "below" if v["auc"] < 0.45 else "level with")
        log.info("   the %d unrated pose(s) score %s your %d '%s' pose(s) "
                 "(AUC %.2f)" % (ev["n_unrated"], verdict, v["n"], label,
                                 v["auc"]))
        if v["auc"] < 0.45:
            worrying.append((label, v["auc"]))
    if worrying:
        log.warning(
            "hitpick learn: the model puts your unrated poses BELOW the ones "
            "you graded %s. If they were really your '%s' ones they should "
            "sit above every graded group, so this list does not look like "
            "it was rated by exception - more like it was rated as far as "
            "you got. Assuming anyway will teach the model the opposite of "
            "what you mean."
            % (", ".join("'%s' (AUC %.2f)" % w for w in worrying), grade))


def _grade_line(counts):
    return ", ".join("%d %s" % (counts[g], g) for g in counts if counts[g]) \
        or "none"


hitpick_learn_assume_desc = CmdDesc(
    optional=[("grade", StringArg), ("forget", BoolArg),
              ("check", BoolArg)],
    synopsis="Treat unrated poses as a grade, for a fully walked list")


hitpick_learn_grade_desc = CmdDesc(
    required=[("grade", StringArg)],
    optional=[("structure", StructureArg)],
    synopsis="Grade a pose good, maybe or bad")


def hitpick_learn_train(session, epochs=120, wait=False):
    """Train the committee on the grades given so far.

    Pure numpy and no ChimeraX objects, so the whole of it goes to the
    worker thread. At a thousand graded poses it is about twenty seconds.
    """
    ensure_ui(session)
    from . import learn, background
    ls = learn.state(session)
    pulled, waiting = ls.pull_viewdock_ratings()
    if pulled:
        session.logger.info("hitpick learn: took %d grade(s) from ViewDock"
                            % pulled)
    if waiting:
        session.logger.warning(
            "hitpick learn: %d rated pose(s) have not been analysed yet and "
            "are being ignored. Run 'hitpick learn collect' to include them."
            % waiting)
    ok, why = ls.learner.can_train()
    if not ok:
        raise UserError(why)

    job = background.claim(session, "training")
    import time
    t0 = time.time()

    def work(j):
        j.report(0, epochs, "training on %d graded pose(s)"
                 % ls.learner.n_graded)
        ls.learner.train(epochs=epochs)
        return True

    def finished(j):
        background.release(session, j)
        if j.error is not None:
            session.logger.warning("hitpick learn: training failed: %s"
                                   % j.error)
        elif j.cancelled:
            session.logger.info("hitpick learn: training cancelled")
        else:
            d = learn.describe(session)
            session.logger.info(
                "hitpick learn: trained on %d graded pose(s) in %.1fs  (%s)"
                % (d["trained_on"], time.time() - t0,
                   ", ".join("%s %d" % (g, n)
                             for g, n in d["grades"].items())))
            # Push the predictions straight into ViewDock: a model whose
            # output you cannot sort is of little use at a party.
            written, why = learn.apply_predictions(session)
            if written:
                session.logger.info(
                    "   %d pose(s) scored in ViewDock - sort by 'HP Pred "
                    "Rank' to work down the predicted-good ones" % written)
            elif why:
                session.logger.info("   %s" % why)
            session.logger.info(
                "   next: hitpick learn next   - the poses worth grading now")
        _refresh_learn_panel(session)

    threaded = job.start(work, on_progress=_job_feedback(session, "learn"),
                         on_done=finished, threaded=(False if wait else None))
    if threaded:
        session.logger.info(
            "hitpick learn: training in the background - about %.0fs at this "
            "many grades. The window stays usable."
            % max(1.0, ls.learner.n_graded * 0.02))
        return {"started": True, "background": True}
    return learn.describe(session)


hitpick_learn_train_desc = CmdDesc(
    optional=[("epochs", IntArg)],
    keyword=[("wait", BoolArg)],
    synopsis="Train the model on the grades given so far")


def hitpick_learn_next(session, count=10, strategy="uncertainty"):
    """The poses worth looking at next."""
    ensure_ui(session)
    from . import learn
    from hitpicking.al import UNCERTAINTY, DISAGREEMENT, PREDICTION
    ls = learn.state(session)
    if ls.learner.committee is None:
        raise UserError("train first: hitpick learn train")
    strat = {"uncertainty": UNCERTAINTY, "disagreement": DISAGREEMENT,
             "prediction": PREDICTION}.get(strategy.lower())
    if strat is None:
        raise UserError("strategy must be uncertainty, disagreement or "
                        "prediction")
    rows = ls.learner.rank_poses(strategy=strat,
                                 exclude_graded=(strat != DISAGREEMENT))
    ls.last_ranked = [k for k, _ in rows]
    if not rows:
        session.logger.info("hitpick learn: nothing left to suggest")
        return []
    header = {UNCERTAINTY: "where the committee disagrees with itself - "
                           "grading these teaches it most",
              DISAGREEMENT: "where the model disagrees with YOUR grade - "
                            "worth a second look",
              PREDICTION: "simply the best predicted"}[strat]
    session.logger.info("<b>hitpick learn: %s</b>" % header, is_html=True)
    for k, p in rows[:count]:
        s = ls.keys.get(k)
        spec = ("#%s" % s.id_string) if s is not None else ""
        extra = ""
        if p["error"] is not None:
            extra = "  you said %s" % ls.learner.grades.label(k)
        session.logger.info(
            "   %-26s predicts %-5s  uncertainty %.4f%s   %s"
            % (k.split("|", 1)[-1][:26], p["grade"], p["uncertainty"],
               extra, spec))
    return [{"key": k, **p} for k, p in rows[:count]]


def hitpick_learn_worklist(session, count=20, strategy="uncertainty",
                           off=False):
    """Put the "look at these next" list into ViewDock and work it down.

    The suggestions are useless in a panel you then have to translate into
    clicks in another window. This sets ViewDock's own table to the queue,
    in the model's order, and a pose leaves the list the moment you grade
    it - so b/m/g walks the batch with no hunting in between.

    `off` hands the table back to ViewDock's own filter. So does finishing
    the batch; the queue is not re-ranked underneath you while you work it,
    because a batch that reshuffles itself is not a batch.
    """
    ensure_ui(session)
    from . import learn, viewdock_link as vl, worklist
    from hitpicking.al import UNCERTAINTY, DISAGREEMENT, PREDICTION

    if off:
        n = worklist.stop(session)
        session.logger.info(
            "hitpick learn: worklist off; ViewDock is showing its own list "
            "again%s" % (" (%d pose(s) were left in the queue)" % n if n
                         else ""))
        _refresh_learn_panel(session)
        return {"stopped": True, "left": n}

    ls = learn.state(session)
    if ls.learner.committee is None:
        raise UserError("train first: hitpick learn train")
    strat = {"uncertainty": UNCERTAINTY, "disagreement": DISAGREEMENT,
             "prediction": PREDICTION}.get(strategy.lower())
    if strat is None:
        raise UserError("strategy must be uncertainty, disagreement or "
                        "prediction")
    tool = vl.find_viewdock(session)
    if tool is None:
        raise UserError(
            "ViewDock is not running, so there is no table to put a "
            "worklist in. The Learn panel still lists the suggestions.")

    rows = ls.learner.rank_poses(strategy=strat,
                                 exclude_graded=(strat != DISAGREEMENT))
    queue = []
    for k, _p in rows[:int(count)]:
        st = ls.keys.get(k)
        if st is not None and getattr(st, "id", None):
            queue.append(st)
    if not queue:
        session.logger.info("hitpick learn: nothing left to suggest")
        return {"queued": 0}

    worklist.stop(session)
    wl = worklist.Worklist(session, tool, queue, label=strategy)
    n = wl.install()
    session.logger.info(
        "hitpick learn: %d pose(s) queued in ViewDock, best first by %s. "
        "Grade with b/m/g and each one drops out as you go. "
        "Stop early with: hitpick learn worklist off" % (n, strategy))
    _refresh_learn_panel(session)
    return {"queued": n, "strategy": strategy}


hitpick_learn_worklist_desc = CmdDesc(
    optional=[("count", IntArg), ("strategy", StringArg), ("off", BoolArg)],
    synopsis="Work the suggestion list down inside ViewDock")


def _score_term_report(session, info, chosen):
    """Print the selection and the evidence behind it."""
    log = session.logger
    rows = info["available"]
    usable = [r for r in rows if r["usable"]]
    if chosen:
        log.info("<b>docking score terms in use</b> (%d of %d slot(s)):"
                 % (len(chosen), info["slots"]), is_html=True)
        for t in chosen:
            r = next((x for x in rows if x["name"] == t), None)
            if r is None:
                log.info("   %s - no loaded pose carries this; "
                         "its column stays empty" % t)
            else:
                log.info("   %-26s mean %8.3f  sd %7.3f  on %d%% of poses"
                         % (t, r["mean"], r["sd"],
                            int(100 * r["coverage"])))
    else:
        log.info("<b>no docking score terms</b> - the model learns from "
                 "geometry and interactions alone.", is_html=True)
    log.info("   %s" % info["why"])
    rejected = [r for r in rows if not r["usable"]]
    spare = [r for r in usable if r["name"] not in chosen]
    if spare:
        log.info("   also available: %s"
                 % ", ".join(r["name"] for r in spare))
    if rejected:
        log.info("   not offered: %s"
                 % "; ".join("%s (%s)" % (r["name"], r["verdict"])
                             for r in rejected))
    log.info("   set with: hitpick learn terms &lt;name&gt;, &lt;name&gt;, ... "
             "| auto | none", is_html=True)


def hitpick_learn_terms(session, terms=None):
    """Choose which of the pose's own score terms the model may learn from.

    The descriptor block ends with seven slots holding the docking
    program's own numbers. Those slots used to be wired to DOCK 3's term
    names, so a mol2 from Glide, Vina, GOLD or rDock put zero in every one
    of them and the model never saw the score its user trusts most.

    With no argument this reports the current selection and everything the
    loaded poses carry, with the junk - row numbers, constants, failed
    terms that came through as 999.999 - named as junk. `auto` picks for
    you; `none` turns the block off.

    Changing the selection rewrites those seven columns on every pose
    already collected, so nothing has to be analysed again. It does
    discard a trained model: its first layer was fitted to columns that
    now hold different quantities. The grades survive - retrain.
    """
    ensure_ui(session)
    from . import learn
    ls = learn.state(session)
    info = ls.suggest_score_terms()

    text = (terms or "").strip()
    if not text:
        _score_term_report(session, info, list(ls.learner.score_terms))
        return {"terms": list(ls.learner.score_terms),
                "available": info["available"], "why": info["why"]}

    low = text.lower()
    if low in ("auto", "default", "detect"):
        wanted = list(info["terms"])
    elif low in ("none", "off", "clear"):
        wanted = []
    else:
        wanted = [t.strip() for t in text.split(",") if t.strip()]
        known = {r["name"]: r for r in info["available"]}
        missing = [t for t in wanted if t not in known]
        if missing:
            raise UserError(
                "no loaded pose carries %s. Run 'hitpick learn terms' with "
                "no argument to see what they do carry."
                % ", ".join(repr(m) for m in missing))
        dead = [t for t in wanted if not known[t]["usable"]]
        if dead:
            session.logger.warning(
                "hitpick learn terms: %s - including it anyway, as you asked"
                % "; ".join("%s is %s (%s)" % (t, known[t]["verdict"],
                                               known[t]["why"]) for t in dead))

    was_trained = ls.learner.committee is not None
    n = ls.set_score_terms(wanted)
    chosen = list(ls.learner.score_terms)
    _score_term_report(session, info, chosen)
    if n:
        session.logger.info(
            "   rewrote the score columns on %d collected pose(s)%s"
            % (n, "; the trained model was discarded - retrain with "
               "'hitpick learn train'" if was_trained else ""))
    _refresh_learn_panel(session)
    return {"terms": chosen, "changed": n, "retrain": bool(was_trained and n)}


hitpick_learn_terms_desc = CmdDesc(
    optional=[("terms", RestOfLine)],
    synopsis="Choose which docking score terms the model learns from")


hitpick_learn_next_desc = CmdDesc(
    optional=[("count", IntArg), ("strategy", StringArg)],
    synopsis="List the poses worth grading next")


def hitpick_learn_status(session):
    ensure_ui(session)
    from . import learn
    d = learn.describe(session)
    log = session.logger
    log.info("<b>hitpick learn</b>", is_html=True)
    log.info("   %d pose(s) featurised, %d graded (%s)"
             % (d["poses"], d["graded"],
                ", ".join("%s %d" % (g, n) for g, n in d["grades"].items())))
    if d["ready"]:
        log.info("   model trained on %d pose(s)" % d["trained_on"])
    elif d["can_train"]:
        log.info("   ready to train: hitpick learn train")
    else:
        log.info("   not ready: %s" % d["why_not"])
    return d


hitpick_learn_status_desc = CmdDesc(
    synopsis="Report the state of the active-learning model")


def hitpick_learn_save(session, path):
    ensure_ui(session)
    from . import learn
    p = learn.save(session, path)
    session.logger.info("hitpick learn: wrote %s" % p)
    return p


hitpick_learn_save_desc = CmdDesc(required=[("path", StringArg)],
                                  synopsis="Save the model and grades")


def hitpick_learn_load(session, path):
    ensure_ui(session)
    from . import learn
    try:
        learn.load(session, path)
    except ValueError as e:
        raise UserError(str(e))
    session.logger.info("hitpick learn: loaded %s" % path)
    return hitpick_learn_status(session)


hitpick_learn_load_desc = CmdDesc(required=[("path", StringArg)],
                                  synopsis="Load a saved model and grades")


def hitpick_clear(session):
    st = get_state(session)
    n = render.clear_groups(session)
    render.clear_atom_labels(session)
    st.result = None
    return n


hitpick_clear_desc = CmdDesc(synopsis="Remove hitpick drawings")


# ------------------------------------------------------------------ register

def register_commands(logger):
    register("hitpick setup", hitpick_setup_desc, hitpick_setup, logger=logger)
    register("hitpick analyze", hitpick_analyze_desc, hitpick_analyze, logger=logger)
    register("hitpick analyse", _analyze_desc(), hitpick_analyze, logger=logger)
    register("hitpick show", hitpick_show_desc, hitpick_show, logger=logger)
    register("hitpick hide", hitpick_hide_desc, hitpick_hide, logger=logger)
    register("hitpick lengths", hitpick_lengths_desc, hitpick_lengths, logger=logger)
    register("hitpick angles", hitpick_angles_desc, hitpick_angles, logger=logger)
    register("hitpick labels", hitpick_labels_desc, hitpick_labels, logger=logger)
    register("hitpick explain", hitpick_explain_desc, hitpick_explain, logger=logger)
    register("hitpick report", hitpick_report_desc, hitpick_report, logger=logger)
    register("hitpick clear", hitpick_clear_desc, hitpick_clear, logger=logger)
    for name, desc, fn in (
            ("panel", hitpick_learn_panel_desc, hitpick_learn_panel),
            ("collect", hitpick_learn_collect_desc, hitpick_learn_collect),
            ("grade", hitpick_learn_grade_desc, hitpick_learn_grade),
            ("assume", hitpick_learn_assume_desc, hitpick_learn_assume),
            ("train", hitpick_learn_train_desc, hitpick_learn_train),
            ("next", hitpick_learn_next_desc, hitpick_learn_next),
            ("worklist", hitpick_learn_worklist_desc,
             hitpick_learn_worklist),
            ("terms", hitpick_learn_terms_desc, hitpick_learn_terms),
            ("status", hitpick_learn_status_desc, hitpick_learn_status),
            ("save", hitpick_learn_save_desc, hitpick_learn_save),
            ("load", hitpick_learn_load_desc, hitpick_learn_load),
            ("cancel", hitpick_learn_cancel_desc,
             hitpick_learn_cancel),
            ("apply", hitpick_learn_apply_desc, hitpick_learn_apply)):
        register("hitpick learn " + name, desc, fn, logger=logger)
    register("hitpick autoexplain", hitpick_autoexplain_desc,
             hitpick_autoexplain, logger=logger)
    register("hitpick scale", hitpick_scale_desc, hitpick_scale, logger=logger)
    register("hitpick clashes", hitpick_clashes_desc, hitpick_clashes,
             logger=logger)
    register("hitpick limits", hitpick_limits_desc, hitpick_limits,
             logger=logger)
    register("hitpick torsions", hitpick_torsions_desc, hitpick_torsions,
             logger=logger)
    register("hitpick hover", hitpick_hover_desc, hitpick_hover, logger=logger)
    register("hitpick torsionprofile", hitpick_torsionprofile_desc,
             hitpick_torsionprofile, logger=logger)
    register("hitpick panel", hitpick_panel_desc, hitpick_panel, logger=logger)
    register("hitpick width", hitpick_width_desc, hitpick_width, logger=logger)
    register("hitpick reveal", hitpick_reveal_desc, hitpick_reveal,
             logger=logger)
    register("hitpick scaleplot", hitpick_scaleplot_desc, hitpick_scaleplot,
             logger=logger)

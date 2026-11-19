"""The `hitpick` commands.

    hitpick setup [receptor #1] [ligand #2]   prepare the binding site
    hitpick analyze [#2]                      analyse one pose and draw it
    hitpick show <types>                      choose which interactions to draw
    hitpick hide <types>
    hitpick lengths true|false
    hitpick labels pKBHX | off                put handbook values on atoms
    hitpick explain <atom-spec>               handbook entry for an atom
    hitpick clear
    hitpick report                            text summary of the current pose
"""

from __future__ import annotations

from chimerax.core.commands import (CmdDesc, register, BoolArg, StringArg,
                                    FloatArg, ModelArg, ModelsArg,
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
                           res.counts, {}, combined=ms, n_receptor=n_rec,
                           strain=strain)

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
        render.label_strain(session, st.result.strain, st.all_atoms)


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
    """Display the residues our interactions point at."""
    ensure_ui(session)
    st = get_state(session)
    if st.result is None:
        raise UserError("nothing analysed yet; run: hitpick analyze")
    n = render.reveal_partners(session, st.result.interactions, st.all_atoms)
    session.logger.info("hitpick: revealed %d interacting atom(s)" % n)
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
        mark = "!" if t.flagged else " "
        cost = ("  %+.1f kcal/mol" % t.energy) if t.energy is not None else ""
        log.info("  %s %-16s %7.1f deg%s   %s - %s"
                 % (mark, t.subtype or t.bond_type, t.angle, cost,
                    ms.atom_label(t.bond[0]), ms.atom_label(t.bond[1])))
        if t.note:
            log.info("      %s" % t.note)
    if not all:
        n_un = sum(1 for t in ts if t.assessment == "unjudged")
        log.info("  %d torsion(s) total; %d flagged. Use 'hitpick torsions "
                 "true' for all of them." % (len(ts), shown))
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
        "group, not the substitution pattern, and substituents move pKBHX by "
        "a log unit or more. A * marks values flagged for review.")
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
    """Print the basicity scale, with the last clicked moiety marked."""
    ensure_ui(session)
    st = get_state(session)
    if st.handbook is None:
        raise UserError("handbook data not built; run tools/build_data.py")
    mids = list(st.last_hit.moiety_ids) if st.last_hit else []
    lines = st.handbook.scale_text(quantity, mids,
                                   same_element_only=not all_elements)
    if not lines:
        session.logger.warning("hitpick: no %s data to plot" % quantity)
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
    for w in r.warnings:
        log.warning("   " + w)
    return _pose_summary(r)


hitpick_report_desc = CmdDesc(synopsis="Text report for the current pose")


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

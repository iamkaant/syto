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
    """Register the mouse mode and hook ViewDock, once, on first use.

    Called from the commands and from the panel - never at startup. Touching
    session.ui during bundle initialisation constructs the mouse-mode manager
    before ChimeraX does, which duplicates the 'mouse hover' trigger and takes
    the Toolbar down with it.
    """
    if getattr(session, "_hitpick_ui_ready", False):
        return
    ui = getattr(session, "ui", None)
    if ui is None or not getattr(ui, "is_gui", False):
        return                      # nogui: nothing to register
    if not hasattr(ui, "mouse_modes"):
        return                      # UI not built yet; try again next time
    session._hitpick_ui_ready = True
    try:
        from .mousemode import register_mouse_mode
        register_mouse_mode(session)
    except Exception as e:
        session.logger.info("hitpick: mouse mode unavailable (%s)" % e)
    try:
        _install_selection_listener(session)
    except Exception as e:
        session.logger.info("hitpick: selection listener unavailable (%s)" % e)
    try:
        from .viewdock_link import autohook
        autohook(session)
    except Exception as e:
        session.logger.info("hitpick: ViewDock link not established (%s)" % e)


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
        _log_hit(session, hit)
        s.notify()

    st._sel_handler = session.triggers.add_handler(SELECTION_CHANGED,
                                                   on_selection)


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
    return st


hitpick_setup_desc = CmdDesc(
    optional=[("receptor", StructureArg), ("ligand", StructureArg)],
    synopsis="Prepare a binding site for hit picking")


# ------------------------------------------------------------------- analyze

def hitpick_analyze(session, ligand=None):
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

    from hitpicking.analysis import PoseResult
    n_rec = st.receptor_model.num_atoms
    st.result = PoseResult(lig.name, res.interactions, res.warnings,
                           res.counts, {}, combined=ms, n_receptor=n_rec)

    _redraw(session, st)
    session.logger.info("hitpick: %s -> %s"
                        % (lig.name, render.legend_text(res.counts)))
    for w in res.warnings:
        session.logger.warning("hitpick: " + w)
    st.notify()
    return st.result


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
    render.draw_interactions(
        session, st.result.interactions, st.all_atoms,
        show_lengths=st.options.show_lengths,
        show_angles=st.options.show_angles)
    if st.options.show_lengths:
        render.label_interactions(session)


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
    return st.options.enabled_nci


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
    return st.options.enabled_nci


hitpick_hide_desc = CmdDesc(optional=[("types", RestOfLine)],
                            synopsis="Disable interaction types")


def hitpick_lengths(session, show=True):
    st = get_state(session)
    st.options.show_lengths = bool(show)
    if not show:
        render.clear_atom_labels(session)
    if st.result is not None:
        _redraw(session, st)
    return st.options.show_lengths


hitpick_lengths_desc = CmdDesc(optional=[("show", BoolArg)],
                               synopsis="Show or hide interaction lengths")


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
        render.clear_atom_labels(session)
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
        "hitpick: labels show the family median and range, not a value for "
        "this exact compound - perception knows the moiety, not the "
        "substitution pattern.")
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
    return hit


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
    for w in r.warnings:
        log.warning("   " + w)
    return r


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

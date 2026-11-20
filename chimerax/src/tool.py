"""The Hit Picking Handbook panel.

HTML rather than native Qt, so the same panel markup and logic serve the web
front end. ViewDock's own table stays native and untouched.

JS -> Python goes through the custom scheme (handle_scheme); Python -> JS
through runJavaScript. Payloads are small JSON, never whole documents.
"""

from __future__ import annotations

import json
import os

from chimerax.core.tools import ToolInstance

try:
    from chimerax.ui import HtmlToolInstance
    _HAVE_HTML = True
except Exception:          # pragma: no cover - nogui
    HtmlToolInstance = object
    _HAVE_HTML = False

from .state import get_state
from . import render, viewdock_link

UI_DIR = os.path.join(os.path.dirname(__file__), "ui")


# The scales the panel offers, in tab order.
SCALE_QUANTITIES = ("pKBHX", "pKBH+")

class HandbookTool(HtmlToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False
    CUSTOM_SCHEME = "hitpick"
    PLACEMENT = "side"
    help = None

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(560, 820),
                         log_errors=True)
        self.display_name = "Hit Picking Handbook"
        # Wide enough that the two-column layout has somewhere to go once the
        # panel is detached; it still collapses to one column when docked.
        from .cmd import ensure_ui
        ensure_ui(session)
        st = get_state(session)
        st.add_listener(self._on_state_changed)
        self._state = st
        self._load_page()

    # ------------------------------------------------------------- page

    def _load_page(self):
        path = os.path.join(UI_DIR, "panel.html")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                html = fh.read()
        else:
            html = "<html><body><p>panel.html is missing</p></body></html>"
        self.html_view.setHtml(html)

    def _js(self, fn, payload):
        try:
            self.html_view.runJavaScript(
                "window.%s(%s)" % (fn, json.dumps(payload)))
        except Exception:
            pass

    # ------------------------------------------------------- state -> UI

    def _on_state_changed(self, st):
        self._js("hpSetState", self._snapshot(st))

    def _snapshot(self, st):
        hb = st.handbook
        snap = {
            "ready": st.ready,
            "options": st.options.to_dict(),
            "handbook": hb.describe() if hb else None,
            "pose": None,
            "hit": None,
            "viewdock": viewdock_link.is_hooked(st.session),
        }
        if st.result is not None:
            r = st.result
            snap["pose"] = {
                "name": r.name,
                "counts": r.counts,
                "warnings": r.warnings,
                "interactions": [i.to_dict() for i in r.interactions[:60]],
                "strain": r.strain.counts if r.strain is not None else None,
                "worst_overlap": (round(r.strain.worst_overlap, 2)
                                  if r.strain is not None else None),
            }
            snap["torsions"] = self._torsion_list(st)
            from hitpicking.strain.torsions import QM_WARN_KCAL
            snap["torsion_warn_kcal"] = QM_WARN_KCAL
        if st.last_hit is not None:
            snap["hit"] = st.last_hit.to_dict()
            snap["torsion"] = self._torsion_profile(st)
            if hb is not None and st.options.show_scale:
                # Both scales, because they answer different questions and
                # the answers do not follow from each other: pKBHX says how
                # well this atom accepts a hydrogen bond, pKBH+ says whether
                # it is holding a proton in the first place. The panel shows
                # whichever the user last looked at; building both here
                # costs a few ms and saves a round trip on every switch.
                # pKBHX is the scale. pKBH+ joins it only when it
                # changes the answer - when this atom is protonated at
                # pH 7.4, so the pKBHX number is for a species that is not
                # in the pocket. Otherwise one tab, and no tab bar at all.
                # `hitpick scale pKBH+` still prints it on demand.
                wanted = {"pKBHX"}
                for a in (snap["hit"].get("annotations") or []):
                    if a.get("changes_the_answer"):
                        wanted.add(a["quantity"])
                scales = []
                for q in SCALE_QUANTITIES:
                    if q not in wanted:
                        continue
                    try:
                        sc = hb.scale(
                            q, st.last_hit.moiety_ids,
                            same_element_only=not st.options.show_angles)
                    except Exception:
                        sc = None
                    if sc:
                        scales.append(sc)
                snap["scales"] = scales
                snap["scale"] = scales[0] if scales else None
        return snap

    # ------------------------------------------------------- UI -> state

    def handle_scheme(self, url):
        from urllib.parse import parse_qs
        path = url.path().lstrip("/")
        query = parse_qs(url.query())

        def one(k, default=None):
            v = query.get(k)
            return v[0] if v else default

        session = self.session
        st = self._state
        try:
            if path == "ready":
                self._on_state_changed(st)
            elif path == "toggle":
                kind = one("kind")
                on = one("on") == "1"
                if kind:
                    if on:
                        st.options.enabled_nci.add(kind)
                    else:
                        st.options.enabled_nci.discard(kind)
                    self._reanalyse()
            elif path == "lengths":
                st.options.show_lengths = one("on") == "1"
                self._reanalyse()
            elif path == "angles":
                st.options.show_angles = one("on") == "1"
                self._reanalyse()
            elif path == "clashes":
                from .cmd import hitpick_clashes
                hitpick_clashes(session, one("on") == "1")
            elif path == "limits":
                from .cmd import hitpick_limits
                hitpick_limits(session, one("on") == "1")
            elif path == "scaleplot":
                from .cmd import hitpick_scaleplot
                hitpick_scaleplot(session, one("on") == "1")
            elif path == "hover":
                from .cmd import hitpick_hover
                hitpick_hover(session, one("on") == "1")
            elif path == "reveal":
                from .cmd import hitpick_reveal
                hitpick_reveal(session, one("on") == "1")
            elif path == "autoexplain":
                from .cmd import hitpick_autoexplain
                hitpick_autoexplain(session, one("on") == "1")
                st.notify()
            elif path == "width":
                from .cmd import hitpick_width
                hitpick_width(session, float(one("value", "0.075")))
                st.notify()
            elif path == "torsion":
                self._pick_torsion(one("bond"))
            elif path == "float":
                self.float_window(one("on") != "0")
            elif path == "labels":
                from .cmd import hitpick_labels
                hitpick_labels(session, one("quantity", "pKBHX"))
            elif path == "analyze":
                from .cmd import hitpick_analyze
                hitpick_analyze(session)
            elif path == "setup":
                from .cmd import hitpick_setup
                hitpick_setup(session)
            elif path == "highlight":
                idx = one("atoms", "")
                atoms = [int(x) for x in idx.split(",") if x.strip().isdigit()]
                render.highlight(session, atoms, st.all_atoms)
            elif path == "explain":
                i = one("atom")
                if i is not None and st.molsys is not None and st.handbook:
                    hit = st.handbook.for_atom(st.molsys, int(i))
                    st.last_hit = hit
                    render.highlight(session, hit.highlight, st.all_atoms)
                    st.notify()
            elif path == "command":
                from chimerax.core.commands import run
                c = one("text")
                if c:
                    run(session, c)
        except Exception as e:
            session.logger.warning("hitpick panel: %s" % e)

    def _pick_torsion(self, bond):
        """Select one torsion from the panel's list.

        The same thing a bond click does, driven from a row instead: the
        bond is selected in 3D, the torsion is numbered on its atoms, and
        the profile below follows it. A list you cannot click through to
        the geometry is just a table.
        """
        st = self._state
        if not bond or st.result is None or st.result.strain is None:
            return
        try:
            want = tuple(sorted(int(x) for x in bond.split(",")))
        except Exception:
            return
        for t in st.result.strain.torsions:
            if tuple(sorted(t.bond)) != want:
                continue
            st.last_torsion = t
            render.show_torsion(self.session, t, st.all_atoms)
            try:
                from chimerax.atomic import Atoms
                self.session.selection.clear()
                Atoms([st.all_atoms[i] for i in t.bond
                       if 0 <= i < len(st.all_atoms)]).selected = True
            except Exception:
                pass
            st.notify()
            return

    def _torsion_list(self, st):
        """Every flagged torsion, worst first, for the panel's list.

        Worst first means by measured cost where there is one. A torsion
        with no QM class has no number and cannot be ranked against one
        that has, so those come after, in bond order.
        """
        if st.result is None or st.result.strain is None:
            return []
        ms = st.molsys
        rows = []
        for t in st.result.strain.torsions:
            if not t.flagged:
                continue
            rows.append({
                "bond": "%d,%d" % t.bond,
                "label": " - ".join(ms.atom_label(a) for a in t.bond)
                         if ms is not None else "",
                "kind": t.subtype or t.bond_type,
                "angle": round(t.angle, 1),
                "assessment": t.assessment,
                "severity": t.severity,
                "energy": (None if t.energy is None else round(t.energy, 2)),
                "note": t.note,
                "selected": (st.last_torsion is not None
                             and st.last_torsion.bond == t.bond),
            })
        rows.sort(key=lambda r: (r["energy"] is None, -(r["energy"] or 0.0),
                                 r["bond"]))
        return rows

    def _torsion_profile(self, st):
        """The QM profile for a clicked bond, or for the worst torsion."""
        if st.result is None or st.result.strain is None:
            return None
        if not st.options.show_scale:
            return None
        from hitpicking.strain import torsions as T
        if T.qm_reference() is None:
            return None
        ts = st.result.strain.torsions
        if not ts:
            return None
        # A torsion belongs to a BOND, so a clicked bond decides it. Driving
        # this from a clicked atom made the profile appear whenever any atom
        # was picked, which is not what was being asked.
        chosen = st.last_torsion
        if chosen is not None and chosen not in ts:
            chosen = None
        if chosen is None:
            flagged = [t for t in ts if t.flagged]
            chosen = flagged[0] if flagged else None
        if chosen is None:
            return None
        try:
            return T.profile_view(chosen.bond_type, chosen.subtype,
                                  chosen.ortho_substituted, chosen.angle)
        except Exception:
            return None

    def float_window(self, floating=True):
        """Detach the panel so it can be widened into two columns."""
        try:
            self.tool_window.floating = bool(floating)
            return True
        except Exception as e:
            self.session.logger.info(
                "hitpick: could not detach the panel (%s). Drag its title bar "
                "out of the ChimeraX window instead." % e)
            return False

    def _reanalyse(self):
        st = self._state
        if st.result is None:
            self._on_state_changed(st)
            return
        from .cmd import hitpick_analyze
        try:
            hitpick_analyze(self.session, st.ligand_model)
        except Exception as e:
            self.session.logger.warning("hitpick: %s" % e)

    # ----------------------------------------------------------- cleanup

    def delete(self):
        try:
            self._state.remove_listener(self._on_state_changed)
        except Exception:
            pass
        super().delete()

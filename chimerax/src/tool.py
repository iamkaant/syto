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
        if st.last_hit is not None:
            snap["hit"] = st.last_hit.to_dict()
            snap["torsion"] = self._torsion_profile(st)
            if hb is not None and st.options.show_scale:
                try:
                    snap["scale"] = hb.scale(
                        "pKBHX", st.last_hit.moiety_ids,
                        same_element_only=not st.options.show_angles)
                except Exception:
                    snap["scale"] = None
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
        chosen = None
        hit = st.last_hit
        if hit is not None and hit.atom is not None:
            # a torsion whose central bond touches the clicked atom
            for t in ts:
                if hit.atom in t.bond:
                    chosen = t
                    break
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

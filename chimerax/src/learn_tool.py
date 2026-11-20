"""The Hit Picking Learn panel.

A separate tool from the handbook, so it only exists when opened, and the
whole active-learning stack loads with it rather than at startup.

It is built around the loop a party actually runs: look at a pose, grade it,
be handed the next one worth looking at. The grade buttons therefore advance
to the next suggestion by default - the model's job is to decide what you
see next, and making the user go back to a list to choose would waste that.
"""

from __future__ import annotations

import json
import os

from chimerax.core.tools import ToolInstance

try:
    from chimerax.ui import HtmlToolInstance
except Exception:          # pragma: no cover - nogui
    HtmlToolInstance = object

UI_DIR = os.path.join(os.path.dirname(__file__), "ui")


def training_curve(learn_state):
    """Loss per epoch for the first committee member.

    A module function rather than a method: it touches nothing on the tool,
    and keeping it free means the panel's data layer can be exercised
    without a Qt window.
    """
    history = learn_state.learner.history
    if not history:
        return None
    pts = [{"epoch": i, "train": round(t, 4),
            "val": None if v is None else round(v, 4)}
           for i, (t, v) in enumerate(history[0])]
    return pts if len(pts) > 1 else None


class LearnTool(HtmlToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False
    CUSTOM_SCHEME = "hplearn"
    PLACEMENT = "side"
    help = None

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(460, 640),
                         log_errors=True)
        self.display_name = "Hit Picking Learn"
        self._strategy = "uncertainty"
        self._auto_advance = True
        from .cmd import ensure_ui
        ensure_ui(session)
        from .state import get_state
        self._hp = get_state(session)
        self._hp.add_listener(self._on_pose_changed)
        self._load_page()

    # ---------------------------------------------------------- plumbing

    def _load_page(self):
        path = os.path.join(UI_DIR, "learn.html")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                html = fh.read()
        else:
            html = "<html><body><p>learn.html is missing</p></body></html>"
        self.html_view.setHtml(html)

    def _js(self, fn, payload):
        try:
            self.html_view.runJavaScript("window.%s(%s)"
                                         % (fn, json.dumps(payload)))
        except Exception:
            pass

    def _on_pose_changed(self, st):
        self.refresh()

    def refresh(self):
        self._js("hpLearnState", self.snapshot())

    # ----------------------------------------------------------- state

    def snapshot(self):
        from . import learn
        session = self.session
        d = learn.describe(session)
        ls = learn.state(session)
        st = self._hp

        current = None
        if st.ligand_model is not None:
            key = ls.key_for(st.ligand_model)
            current = {
                "name": str(st.ligand_model.name).strip(),
                "spec": "#%s" % st.ligand_model.id_string,
                "key": key,
                "grade": ls.learner.grades.label(key),
                "featurised": key in ls.learner.features,
            }
            if st.result is not None:
                current["summary"] = st.result.summary()
                if st.result.strain is not None:
                    current["worst_overlap"] = round(
                        st.result.strain.worst_overlap, 2)
                current["good"] = sum(1 for i in st.result.interactions
                                      if i.quality >= 0.55)

        suggestions = []
        if ls.learner.committee is not None:
            from hitpicking.al import UNCERTAINTY, DISAGREEMENT, PREDICTION
            strat = {"uncertainty": UNCERTAINTY,
                     "disagreement": DISAGREEMENT,
                     "prediction": PREDICTION}[self._strategy]
            try:
                rows = ls.learner.rank_poses(
                    strategy=strat, exclude_graded=(strat != DISAGREEMENT))
            except Exception:
                rows = []
            for k, p in rows[:25]:
                s = ls.keys.get(k)
                suggestions.append({
                    "key": k,
                    "name": k.split("|", 1)[-1][:34],
                    "spec": ("#%s" % s.id_string) if s is not None else "",
                    "grade": p["grade"],
                    "uncertainty": round(p["uncertainty"], 4),
                    "yours": ls.learner.grades.label(k),
                    "error": p["error"],
                })

        from . import background
        job = background.current(session)
        running = None
        if job is not None:
            n, total, msg = job.progress
            running = {"title": job.title, "done": n, "total": total,
                       "message": msg,
                       "percent": int(100.0 * n / total) if total else 0}

        from . import worklist as _wl
        wl = _wl.current(self.session)
        return {
            "stats": d,
            "running": running,
            "worklist": wl.describe() if wl is not None else None,
            "options": list(ls.learner.options),
            "current": current,
            "strategy": self._strategy,
            "auto_advance": self._auto_advance,
            "suggestions": suggestions,
            "curve": training_curve(ls),
        }

    # ------------------------------------------------------ UI actions

    def handle_scheme(self, url):
        from urllib.parse import parse_qs
        from chimerax.core.commands import run
        path = url.path().lstrip("/")
        q = parse_qs(url.query())

        def one(k, default=None):
            v = q.get(k)
            return v[0] if v else default

        session = self.session
        try:
            if path == "ready":
                self.refresh()
            elif path == "collect":
                run(session, "hitpick learn collect")
                self.refresh()
            elif path == "train":
                run(session, "hitpick learn train")
                self.refresh()
            elif path == "cancel":
                run(session, "hitpick learn cancel")
                self.refresh()
            elif path == "grade":
                self._grade(one("value"))
            elif path == "worklist":
                from .cmd import hitpick_learn_worklist
                if one("off") == "1":
                    hitpick_learn_worklist(self.session, off=True)
                else:
                    hitpick_learn_worklist(
                        self.session, count=int(one("count", "20") or 20),
                        strategy=self._strategy)
                self.refresh()
            elif path == "assume":
                from .cmd import hitpick_learn_assume
                if one("forget") == "1":
                    hitpick_learn_assume(self.session, forget=True)
                else:
                    hitpick_learn_assume(self.session, one("value", "good"))
                self.refresh()
            elif path == "strategy":
                self._strategy = one("value", "uncertainty")
                self.refresh()
            elif path == "autoadvance":
                self._auto_advance = one("on") == "1"
                self.refresh()
            elif path == "show":
                self._show(one("key"))
            elif path == "next":
                self._advance()
            elif path == "save":
                self._save_dialog()
            elif path == "load":
                self._load_dialog()
        except Exception as e:
            session.logger.warning("hitpick learn panel: %s" % e)
            self.refresh()

    def _grade(self, value):
        from chimerax.core.commands import run
        if not value:
            return
        st = self._hp
        if st.ligand_model is None:
            self.session.logger.warning(
                "hitpick learn: no pose displayed to grade")
            return
        run(self.session, "hitpick learn grade %s" % value)
        if self._auto_advance:
            self._advance()
        else:
            self.refresh()

    def _advance(self):
        """Show the next pose worth grading."""
        snap = self.snapshot()
        for s in snap["suggestions"]:
            if s["key"] != (snap["current"] or {}).get("key"):
                self._show(s["key"])
                return
        self.session.logger.info(
            "hitpick learn: nothing further to suggest - train again after "
            "grading a few more")
        self.refresh()

    def _show(self, key):
        """Display one pose, keeping ViewDock's table in step."""
        from chimerax.core.commands import run
        from . import learn, viewdock_link as vl
        if not key:
            return
        ls = learn.state(self.session)
        s = ls.keys.get(key)
        if s is None:
            return
        tool = vl.find_viewdock(self.session)
        if tool is not None:
            try:
                # let ViewDock drive, so its own display logic and our
                # pose-change hook both run exactly as if a row was clicked
                tool.struct_table.selected = [s]
                self.refresh()
                return
            except Exception:
                pass
        run(self.session, "hitpick analyze #%s" % s.id_string)
        self.refresh()

    def _save_dialog(self):
        from Qt.QtWidgets import QFileDialog
        from chimerax.core.commands import run
        path, _ = QFileDialog.getSaveFileName(
            None, "Save the party's grades and model", "party.json",
            "JSON (*.json)")
        if path:
            run(self.session, "hitpick learn save %s" % path)

    def _load_dialog(self):
        from Qt.QtWidgets import QFileDialog
        from chimerax.core.commands import run
        path, _ = QFileDialog.getOpenFileName(
            None, "Load grades and model", "", "JSON (*.json)")
        if path:
            run(self.session, "hitpick learn load %s" % path)
            self.refresh()

    # ----------------------------------------------------------- cleanup

    def delete(self):
        try:
            self._hp.remove_listener(self._on_pose_changed)
        except Exception:
            pass
        super().delete()

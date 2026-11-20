#!/usr/bin/env python
"""Pack the hitpicking core into a zip the browser worker imports.

The web page runs the SAME Python as the ChimeraX bundle - not a port, not
a reimplementation - so there is exactly one place where an interaction is
decided. This copies `core/hitpicking` into `web/vendor/hitpicking.zip`,
which Pyodide unpacks onto its filesystem.

The compiled handbook travels with it, so the page needs no network.
"""
from __future__ import annotations

import hashlib
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "core", "hitpicking")
OUT = os.path.join(ROOT, "web", "vendor", "hitpicking.zip")
UI_SRC = os.path.join(ROOT, "ui")
UI_DST = os.path.join(ROOT, "web", "ui")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
KEEP_EXT = {".py", ".json", ".gz"}


def main():
    if not os.path.isdir(SRC):
        print("no core at %s" % SRC)
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    files = []
    for base, dirs, names in os.walk(SRC):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for n in sorted(names):
            if os.path.splitext(n)[1] not in KEEP_EXT:
                continue
            full = os.path.join(base, n)
            files.append((full, os.path.relpath(full, os.path.dirname(SRC))))

    digest = hashlib.blake2b(digest_size=8)
    # deterministic: fixed order, fixed timestamps, so an unchanged core
    # produces a byte-identical zip and the page's cache stays valid
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for full, arc in files:
            with open(full, "rb") as fh:
                data = fh.read()
            digest.update(arc.encode("utf-8"))
            digest.update(data)
            info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)

    # The shared panels are vendored INTO web/ rather than referenced at
    # ../ui, so the served directory is self-contained: a web server rooted
    # at web/ cannot reach a sibling, and neither can a copy someone zips
    # up and mails to a collaborator.
    import shutil
    n_ui = 0
    if os.path.isdir(UI_SRC):
        if os.path.isdir(UI_DST):
            shutil.rmtree(UI_DST)
        os.makedirs(UI_DST, exist_ok=True)
        for n in sorted(os.listdir(UI_SRC)):
            if n.startswith(".") or os.path.isdir(os.path.join(UI_SRC, n)):
                continue
            shutil.copy2(os.path.join(UI_SRC, n), os.path.join(UI_DST, n))
            n_ui += 1
        # The handbook panel still lives beside the bundle, because that is
        # the host that loads it from disk. It is written against a host
        # abstraction (window.hpHost) rather than against ChimeraX, so the
        # web page runs the same file in an iframe - which is the only way
        # the two front ends can be guaranteed to explain an atom
        # identically: there is one renderer, not two.
        panel = os.path.join(ROOT, "chimerax", "src", "ui", "panel.html")
        if os.path.exists(panel):
            shutil.copy2(panel, os.path.join(UI_DST, "panel.html"))
            n_ui += 1

    size = os.path.getsize(OUT)
    print("%s: %d files, %.1f KB, core hash %s"
          % (os.path.relpath(OUT, ROOT), len(files), size / 1024.0,
             digest.hexdigest()))
    print("web/ui: %d shared panel file(s)" % n_ui)
    return 0


if __name__ == "__main__":
    sys.exit(main())

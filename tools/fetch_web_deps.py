#!/usr/bin/env python
"""Fetch the browser runtime the web page needs, into web/vendor/.

That directory is gitignored: it is 17 MB of someone else's build product
and does not belong in the history. This script puts it back, pinned to an
exact version, so a fresh clone can run the page offline.

Pyodide ships a `core` tarball without any packages, which is what we
want: numpy is the only one needed and it comes as a single wheel. The
alternative is the full 337 MB distribution.

Usage:  python tools/fetch_web_deps.py [--force]
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "web", "vendor", "pyodide")

# Pinned. Bumping this is a deliberate act: the numpy ABI tag below has to
# move with it, and the parity check has to be re-run, because the whole
# point is that the browser runs the same numpy the bundle does.
PYODIDE_VERSION = "314.0.8"
NUMPY_WHEEL = "numpy-2.4.6-cp314-cp314-pyemscripten_2026_0_wasm32.whl"

CORE_URL = ("https://github.com/pyodide/pyodide/releases/download/"
            "%s/pyodide-core-%s.tar.bz2" % (PYODIDE_VERSION, PYODIDE_VERSION))
WHEEL_URL = ("https://cdn.jsdelivr.net/pyodide/v%s/full/%s"
             % (PYODIDE_VERSION, NUMPY_WHEEL))

# A generic client name: the fetch must not carry anything about whoever
# is running it.
HEADERS = {"User-Agent": "hitpicking-build"}


def get(url, dest):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as r, \
            open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)
    return os.path.getsize(dest)


def main(argv):
    force = "--force" in argv
    marker = os.path.join(VENDOR, "pyodide.mjs")
    if os.path.exists(marker) and not force:
        print("web/vendor/pyodide already present (--force to refetch)")
        return 0

    os.makedirs(VENDOR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tarball = os.path.join(tmp, "core.tar.bz2")
        print("fetching pyodide %s ..." % PYODIDE_VERSION)
        n = get(CORE_URL, tarball)
        print("  %.1f MB" % (n / 1e6))
        # extract into the temp dir, then move only the files we want, so a
        # surprising path in the archive cannot land outside web/vendor
        with tarfile.open(tarball, "r:bz2") as tf:
            safe = [m for m in tf.getmembers()
                    if m.isfile() and not m.name.startswith(("/", ".."))
                    and ".." not in m.name.split("/")]
            tf.extractall(tmp, members=safe)
        root = os.path.join(tmp, "pyodide")
        if not os.path.isdir(root):
            print("unexpected archive layout; nothing extracted")
            return 1
        for n in sorted(os.listdir(root)):
            shutil.move(os.path.join(root, n), os.path.join(VENDOR, n))

    print("fetching %s ..." % NUMPY_WHEEL)
    n = get(WHEEL_URL, os.path.join(VENDOR, NUMPY_WHEEL))
    print("  %.1f MB" % (n / 1e6))

    total = sum(os.path.getsize(os.path.join(VENDOR, f))
                for f in os.listdir(VENDOR)
                if os.path.isfile(os.path.join(VENDOR, f)))
    print("web/vendor/pyodide: %.1f MB, pyodide %s + numpy"
          % (total / 1e6, PYODIDE_VERSION))
    print("now run: python tools/build_web_core.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Build and verify the ChimeraX wheel that goes to the Toolshed.

`devel build` already makes a wheel. What it does not do is tell you
whether the wheel is the thing you meant to publish: it will happily
package a stale vendored core, a handbook.json from a data build you
never re-ran, a dev version number, or a tree with uncommitted edits in
it. A wheel on the Toolshed is public and its version can never be
reused, so every one of those is a mistake you cannot take back.

So this does the whole release path in order and refuses at the first
thing that is wrong:

  1. the version is a real release version, and not one already published
  2. the working tree is clean, so the wheel can be rebuilt from a commit
  3. the handbook data builds and validates
  4. the shared core is re-vendored, and verified to match core/
  5. ChimeraX builds the wheel
  6. the wheel is opened and checked against the source tree, file by file
  7. the vendored core inside the wheel is imported, in a fresh process

Usage:
    python tools/release_bundle.py                 # build 0.1.0 (pyproject)
    python tools/release_bundle.py --version 0.2.0 # and write it back
    python tools/release_bundle.py --check         # verify, do not build
    python tools/release_bundle.py --allow-dirty   # a test build

The wheel lands in dist/ at the top of the repo. Submitting it is a
manual step and deliberately stays one - see the instructions printed at
the end, or README.md.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUNDLE = os.path.join(ROOT, "chimerax")
SRC = os.path.join(BUNDLE, "src")
TOML = os.path.join(BUNDLE, "pyproject.toml")
DIST = os.path.join(ROOT, "dist")

PKG = "chimerax/hitpickingtool"       # where src/ lands inside the wheel
DEFAULT_APP = "/Applications/ChimeraX-1.12.app"
TOOLSHED = "https://cxtoolshed.rbvi.ucsf.edu"
BUNDLE_NAME = "ChimeraX-HitPickingTool"

# A release version: 1.2, 1.2.3, 1.2.3rc1, 1.2.3.post1. Not .dev, not a
# local version (+something) - the Toolshed takes neither, and a dev
# stamp left over from tools/install_bundle.sh is exactly the accident
# this is here to catch.
RELEASE_RE = re.compile(r"^\d+(\.\d+)+((a|b|rc)\d+|\.post\d+)?$")

SKIP_NAMES = {".DS_Store"}
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


class Failed(Exception):
    pass


def say(msg):
    print(msg)


def ok(msg):
    print("  OK   " + msg)


def run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


# ----------------------------------------------------------------- steps

def read_version():
    with open(TOML, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r'^version = "(.*)"', line)
            if m:
                return m.group(1)
    raise Failed("no version in %s" % TOML)


def write_version(version):
    with open(TOML, encoding="utf-8") as fh:
        text = fh.read()
    new = re.sub(r'^version = ".*"', 'version = "%s"' % version, text,
                 count=1, flags=re.M)
    if new == text:
        return False
    with open(TOML, "w", encoding="utf-8") as fh:
        fh.write(new)
    return True


def check_version(version):
    if not RELEASE_RE.match(version):
        raise Failed(
            "%r is not a release version. The Toolshed publishes what you "
            "give it under that exact number and the number can never be "
            "reused, so a .dev or +local build must not be submitted. "
            "A dev stamp here usually means tools/install_bundle.sh was "
            "interrupted before it restored pyproject.toml." % version)
    ok("version %s is a release version" % version)
    return version


def check_not_published(version, offline):
    """Has this version already gone out?

    The Toolshed publishes one wheel per version and a version can never
    be replaced, so submitting a number that is already there is a dead
    end you find out about after uploading. Its catalog is a plain JSON
    list of every release - the same one ChimeraX itself reads - so the
    answer is one request away.
    """
    if offline:
        say("  --  skipped the published-version check (--offline)")
        return
    # The endpoint chimerax.core.toolshed.available reads. No uuid is
    # sent: it is ChimeraX's install id and has nothing to do with this.
    url = "%s/bundle/?format_version=1" % TOOLSHED
    try:
        import json
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent":
                                                   "python-urllib"})
        with urllib.request.urlopen(req, timeout=30) as fh:
            catalog = json.loads(fh.read().decode("utf-8", "replace"))
    except Exception as e:
        say("  --  could not ask the Toolshed what is published (%s); "
            "check by hand before submitting" % e)
        return

    def same(name):
        return (name or "").replace("_", "-").lower() == BUNDLE_NAME.lower()

    seen = []
    for rel in catalog:
        if not isinstance(rel, dict):
            continue
        if not same(rel.get("bundle_name")):
            continue
        for info in (rel.get("bundle") or {}).values():
            v = (info or {}).get("version")
            if v:
                seen.append(str(v))
    if version in seen:
        raise Failed("%s is already on the Toolshed (versions there: %s). "
                     "Bump the version: a published one cannot be replaced."
                     % (version, ", ".join(sorted(set(seen)))))
    if seen:
        ok("not yet published; %s is already up there"
           % ", ".join(sorted(set(seen))))
    else:
        ok("nothing by this name on the Toolshed yet - this would be the "
           "first submission, which waits for staff approval")


def check_clean_tree(allow_dirty):
    r = run(["git", "-C", ROOT, "status", "--porcelain"])
    if r.returncode != 0:
        say("  --  not a git checkout; cannot tell what this wheel was "
            "built from")
        return None
    dirty = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if dirty and not allow_dirty:
        raise Failed(
            "the working tree has uncommitted changes, so this wheel could "
            "not be rebuilt from any commit:\n    "
            + "\n    ".join(dirty[:12])
            + "\n  Commit them, or pass --allow-dirty for a test build.")
    head = run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).stdout.strip()
    if dirty:
        say("  --  building from a DIRTY tree at %s (--allow-dirty)" % head)
    else:
        ok("clean tree at %s" % head)
    return head


def python_for_data():
    """A Python with PyYAML, for the handbook build."""
    cand = os.environ.get("HITPICK_PYTHON")
    if cand and os.path.exists(cand):
        return cand
    guess = os.path.expanduser("~/miniforge3/envs/admet_ai/bin/python")
    if os.path.exists(guess):
        return guess
    return sys.executable


def build_data(strict):
    py = python_for_data()
    cmd = [py, os.path.join(HERE, "build_data.py"), "--quiet"]
    if strict:
        cmd.append("--release")
    r = run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise Failed("the handbook data did not build:\n"
                     + (r.stdout + r.stderr)[-2000:])
    # Say out loud what state the curation is in. Not a failure - the
    # data is honestly still being curated - but it ships in the wheel
    # and whoever submits it should know what they are publishing.
    r = run([py, os.path.join(HERE, "build_data.py"), "--check"], cwd=ROOT)
    for line in r.stdout.strip().splitlines():
        say("       " + line)
    ok("handbook data built and validated")


def vendor_core():
    py = python_for_data()
    v = os.path.join(HERE, "vendor_core.py")
    r = run([py, v], cwd=ROOT)
    if r.returncode != 0:
        raise Failed("vendoring the core failed:\n" + r.stdout + r.stderr)
    r = run([py, v, "--check"], cwd=ROOT)
    if r.returncode != 0:
        raise Failed("the vendored core does not match core/ even after "
                     "copying it:\n" + r.stdout + r.stderr)
    ok("core vendored into the bundle and verified against core/")


def source_files():
    """Every file in src/, as the path it should have inside the wheel."""
    out = set()
    for base, dirs, names in os.walk(SRC):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in names:
            if n in SKIP_NAMES or n.endswith((".pyc", ".pyo")):
                continue
            rel = os.path.relpath(os.path.join(base, n), SRC)
            out.add(PKG + "/" + rel.replace(os.sep, "/"))
    return out


def build_wheel(app, version):
    cx = os.path.join(app, "Contents", "bin", "ChimeraX")
    if not os.path.exists(cx):          # Linux / Windows layouts
        for cand in (os.path.join(app, "bin", "ChimeraX"), app):
            if os.path.exists(cand) and os.access(cand, os.X_OK):
                cx = cand
                break
    if not (os.path.exists(cx) and os.access(cx, os.X_OK)):
        raise Failed("no ChimeraX at %s. The wheel has to be built by "
                     "ChimeraX's own bundle builder - there is no way to "
                     "produce a valid bundle without it. Pass --app." % app)

    for d in ("build", "dist"):
        shutil.rmtree(os.path.join(BUNDLE, d), ignore_errors=True)
    for n in os.listdir(BUNDLE):
        if n.endswith(".egg-info"):
            shutil.rmtree(os.path.join(BUNDLE, n), ignore_errors=True)

    r = run([cx, "--nogui", "--silent", "--exit",
             "--cmd", "devel build %s ; exit" % BUNDLE], timeout=1800)
    out = r.stdout + r.stderr
    wheels = []
    dist = os.path.join(BUNDLE, "dist")
    if os.path.isdir(dist):
        wheels = [os.path.join(dist, n) for n in os.listdir(dist)
                  if n.endswith(".whl")]
    if not wheels:
        raise Failed("devel build produced no wheel:\n" + out[-3000:])
    if len(wheels) > 1:
        raise Failed("more than one wheel in %s: %s" % (dist, wheels))
    wheel = wheels[0]
    want = "-%s-" % version
    if want not in os.path.basename(wheel):
        raise Failed("the wheel is %s but the version asked for was %s"
                     % (os.path.basename(wheel), version))
    ok("ChimeraX built %s" % os.path.basename(wheel))
    return wheel


def check_wheel(wheel, version):
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
        meta = None
        for n in names:
            if n.endswith(".dist-info/METADATA"):
                meta = z.read(n).decode("utf-8", "replace")
        if meta is None:
            raise Failed("the wheel has no METADATA")

    missing = sorted(source_files() - names)
    if missing:
        raise Failed(
            "%d file(s) in chimerax/src are NOT in the wheel - they would "
            "simply be absent for everyone who installs it:\n    %s"
            % (len(missing), "\n    ".join(missing[:20])))
    ok("every file in chimerax/src is in the wheel (%d)" % len(source_files()))

    junk = sorted(n for n in names
                  if "__pycache__" in n or n.endswith((".pyc", ".pyo"))
                  or os.path.basename(n) in SKIP_NAMES)
    if junk:
        raise Failed("the wheel carries build junk: %s" % junk[:10])
    ok("no caches or editor droppings in it")

    # The data the handbook needs at runtime. A bundle that installs and
    # then says "handbook data not built" is worse than one that fails.
    for want in ("_vendor/hitpicking/data/handbook.json",
                 "_vendor/hitpicking/data/torsion_qm.json",
                 "ui/panel.html", "ui/learn.html"):
        if PKG + "/" + want not in names:
            raise Failed("the wheel is missing %s" % want)
    ok("handbook data, torsion profiles and both panels are inside")

    for line, why in (
            ("Name: ChimeraX-HitPickingTool", "the Toolshed keys on the name"),
            ("Version: %s" % version, "the version must match"),
            ("Classifier: Framework :: ChimeraX",
             "without it the Toolshed does not see a bundle"),
            ("Requires-Dist: ChimeraX-Core", "the one real dependency")):
        if line not in meta:
            raise Failed("METADATA has no %r - %s" % (line, why))
    ok("METADATA names, versions and classifies it as a ChimeraX bundle")

    cmds = len([ln for ln in meta.splitlines()
                if ln.startswith("Classifier: ChimeraX :: Command")])
    tools = len([ln for ln in meta.splitlines()
                 if ln.startswith("Classifier: ChimeraX :: Tool")])
    if cmds < 10 or tools < 2:
        raise Failed("only %d command(s) and %d tool(s) are declared; the "
                     "bundle registers far more than that, so something in "
                     "pyproject.toml was dropped" % (cmds, tools))
    ok("%d commands and %d tools are declared" % (cmds, tools))
    return meta


def check_imports(wheel):
    """Unpack the wheel and import the core out of it, in a fresh process.

    Every check above reads the wheel as a list of names. This one runs
    what is in it: a syntax error or a missing module in the vendored
    core gets through every other check and fails at the user's end.
    """
    import tempfile
    with tempfile.TemporaryDirectory(prefix="hp-wheel-") as tmp:
        with zipfile.ZipFile(wheel) as z:
            z.extractall(tmp)
        vendor = os.path.join(tmp, PKG, "_vendor")
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import hitpicking\n"
            "from hitpicking.analysis import Options, ReceptorContext\n"
            "from hitpicking.strain import analyze_strain\n"
            "from hitpicking.strain.torsions import qm_reference, QM_WARN_KCAL\n"
            "from hitpicking.al import Learner\n"
            "from hitpicking.handbook import Handbook\n"
            "hb = Handbook.load()\n"
            "print('core', hitpicking.__version__, 'features',"
            " hitpicking.FEATURE_VERSION)\n"
            "print('handbook', hb.content_hash, len(hb.entries), 'entries')\n"
            "print('torsion classes', len((qm_reference() or {}).get("
            "'classes', {})), 'warn at', QM_WARN_KCAL)\n"
        ) % vendor
        r = run([sys.executable, "-I", "-c", code], cwd=tmp)
        if r.returncode != 0:
            raise Failed("the core inside the wheel does not import:\n"
                         + (r.stdout + r.stderr)[-2000:])
        for line in r.stdout.strip().splitlines():
            say("       " + line)
    ok("the vendored core imports and the handbook loads, from the wheel")


def stage(wheel):
    os.makedirs(DIST, exist_ok=True)
    dest = os.path.join(DIST, os.path.basename(wheel))
    shutil.copy2(wheel, dest)
    return dest


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", help="release version; written into "
                                      "chimerax/pyproject.toml")
    ap.add_argument("--app", default=DEFAULT_APP, help="ChimeraX.app to "
                    "build with (default: %s)" % DEFAULT_APP)
    ap.add_argument("--check", action="store_true",
                    help="run the pre-build checks only")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build from an uncommitted tree (test builds)")
    ap.add_argument("--strict-data", action="store_true",
                    help="fail if any handbook value is still unverified")
    ap.add_argument("--offline", action="store_true",
                    help="do not ask the Toolshed what is already published")
    a = ap.parse_args()

    try:
        version = check_version(a.version or read_version())
        if a.version and write_version(a.version):
            say("       wrote version %s into chimerax/pyproject.toml "
                "- commit it" % a.version)
        check_not_published(version, a.offline)
        head = check_clean_tree(a.allow_dirty)
        build_data(a.strict_data)
        vendor_core()
        if a.check:
            say("\nchecks passed; nothing built (--check)")
            return 0
        wheel = build_wheel(a.app, version)
        check_wheel(wheel, version)
        check_imports(wheel)
        dest = stage(wheel)
    except Failed as e:
        print("\nFAILED: %s" % e)
        return 1

    size = os.path.getsize(dest) / 1024.0
    say("")
    say("  %s  (%.0f KB)%s" % (dest, size, ", built from %s" % head if head else ""))
    say("")
    say("To publish it:")
    say("  1. sign in at %s (Google sign-in only)" % TOOLSHED)
    say("  2. 'Submit a Bundle', upload the wheel above")
    say("  3. the FIRST submission waits for ChimeraX staff to approve it;")
    say("     later ones from the same account appear immediately")
    say("")
    say("To try it first, in ChimeraX:")
    say("  toolshed install %s" % dest)
    say("  (or tools/install_bundle.sh for a dev install)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# Build and (re)install the ChimeraX bundle during development.
#
# ----------------------------------------------------------------------------
# NEVER use `pip install --target "<ChimeraX site-packages>"` here.
#
# This bundle's top-level directory is `chimerax/`, the same namespace package
# that holds every other ChimeraX bundle. pip's --target mode with --upgrade
# DELETES a colliding directory in the target before installing, so a single
# such command wipes chimerax/isolde, chimerax/clipper and every other
# user-installed bundle, leaving their dist-info behind so ChimeraX keeps
# trying to import modules that no longer exist. That happened once already.
#
# Use ChimeraX's own `devel install`, which knows where bundles belong and
# installs only this bundle's own files.
# ----------------------------------------------------------------------------
#
# `devel install` hands the wheel to pip, which SKIPS installing when the
# version is unchanged - during development it never changes, so edits
# silently have no effect. So each build gets a unique dev version, written
# into pyproject.toml just for the build and restored afterwards.
#
# Usage:  tools/install_bundle.sh [/path/to/ChimeraX.app]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${1:-/Applications/ChimeraX-1.12.app}"
CX="$APP/Contents/bin/ChimeraX"
PY="$APP/Contents/bin/python3.11"
TOML="$ROOT/chimerax/pyproject.toml"

if [ ! -x "$CX" ]; then
  echo "ChimeraX not found at $CX" >&2
  exit 1
fi

PYBIN="${HITPICK_PYTHON:-$PY}"

echo "==> building handbook data"
"$PYBIN" "$ROOT/tools/build_data.py" --quiet || {
  echo "    (needs PyYAML; set HITPICK_PYTHON to a python that has it)" >&2
}

echo "==> vendoring the shared core into the bundle"
"$PYBIN" "$ROOT/tools/vendor_core.py"

BASE_VERSION="$(grep -m1 '^version = ' "$TOML" | sed -E 's/version = "(.*)"/\1/')"
DEV_VERSION="${BASE_VERSION%%.dev*}.dev$(date +%s)"

restore_version() {
  if [ -f "$TOML.orig" ]; then
    mv -f "$TOML.orig" "$TOML"
  fi
}
trap restore_version EXIT

cp "$TOML" "$TOML.orig"
sed -i '' -E "s/^version = \".*\"/version = \"$DEV_VERSION\"/" "$TOML"

echo "==> building and installing $DEV_VERSION"
rm -rf "$ROOT/chimerax/build" "$ROOT/chimerax/dist" "$ROOT/chimerax"/*.egg-info
OUT="$("$CX" --nogui --silent --exit --cmd "devel install $ROOT/chimerax ; exit" 2>&1)" || true
echo "$OUT" | grep -iE "^error|Traceback|is in use" | head -5 || true

restore_version
trap - EXIT

SITE="$HOME/Library/Application Support/ChimeraX/${APP##*ChimeraX-}"
SITE="${SITE%.app}/lib/python/site-packages"
if [ -d "$SITE/chimerax" ]; then
  echo "==> sibling bundles still present:"
  ls "$SITE/chimerax" | sed 's/^/      /'
fi

# --silent swallows the usage text, so this reads the real log. It used to
# grep for the package name, which stopped appearing when the bundle was
# renamed - a check that silently passes whatever happens is not a check.
echo "==> verifying the command registers"
OUT="$("$CX" --nogui --exit --cmd "usage hitpick analyze ; exit" 2>&1)"
if echo "$OUT" | grep -q "'hitpick analyze'"; then
  echo "    hitpick analyze is registered"
else
  echo "    FAILED: hitpick analyze did not register"
  echo "$OUT" | grep -iE "error|No module named|Traceback" | head -5
fi

echo "done. In ChimeraX:  hitpick setup"

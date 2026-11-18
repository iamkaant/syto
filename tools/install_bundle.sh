#!/usr/bin/env bash
# Build and (re)install the ChimeraX bundle during development.
#
# `devel install` alone is not enough: it builds a wheel and hands it to pip,
# which SKIPS the install when the version has not changed. During development
# the version never changes, so edits silently do not take effect. This script
# forces the reinstall.
#
# Usage:  tools/install_bundle.sh [/path/to/ChimeraX.app]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${1:-/Applications/ChimeraX-1.12.app}"
CX="$APP/Contents/bin/ChimeraX"
PY="$APP/Contents/bin/python3.11"

if [ ! -x "$CX" ]; then
  echo "ChimeraX not found at $CX" >&2
  exit 1
fi

PYBIN="${HITPICK_PYTHON:-$PY}"

# ChimeraX user site-packages lives under the app's minor version
CX_VER="$(basename "$APP" | sed -E 's/^ChimeraX-?//; s/\.app$//' | cut -d. -f1,2)"
if [ -z "$CX_VER" ]; then CX_VER="1.12"; fi

echo "==> building handbook data"
"$PYBIN" "$ROOT/tools/build_data.py" --quiet || {
  echo "   (handbook build needs PyYAML; skipping)" >&2
}

echo "==> vendoring the shared core into the bundle"
"$PYBIN" "$ROOT/tools/vendor_core.py"

echo "==> building the wheel"
rm -rf "$ROOT/chimerax/build" "$ROOT/chimerax/dist" "$ROOT/chimerax"/*.egg-info
"$CX" --nogui --silent --exit --cmd "devel build $ROOT/chimerax ; exit" >/dev/null 2>&1

WHEEL="$(ls -t "$ROOT"/chimerax/dist/*.whl 2>/dev/null | head -1)"
if [ -z "$WHEEL" ]; then
  echo "no wheel produced" >&2
  exit 1
fi
echo "    $WHEEL"

# `toolshed install` defers when any bundle is in use ("ChimeraX must be
# restarted"), which never completes in a --exit session. Installing the wheel
# straight into ChimeraX's user site-packages is reliable and is what the
# toolshed would have done anyway.
SITE="$("$CX" --nogui --silent --exit --cmd \
  "exit" 2>/dev/null >/dev/null; echo "")"
SITE="$HOME/Library/Application Support/ChimeraX/$CX_VER/lib/python/site-packages"

echo "==> installing into $SITE"
"$PY" -m pip install --quiet --force-reinstall --no-deps --upgrade \
  --target "$SITE" "$WHEEL" 2>&1 | grep -viE "notice|^\s*$" || true

echo "==> verifying"
"$CX" --nogui --silent --exit --cmd "usage hitpick setup ; exit" 2>&1 \
  | grep -iE "hitpick|error" | head -3 || true
echo "done. In ChimeraX:  hitpick setup"

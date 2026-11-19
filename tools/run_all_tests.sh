#!/usr/bin/env bash
# Everything that can be checked without a GUI.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CX="${CHIMERAX_APP:-/Applications/ChimeraX-1.12.app}/Contents/bin/ChimeraX"
PY="${HITPICK_PYTHON:-/Users/ak87/miniforge3/envs/admet_ai/bin/python}"
fail=0

# Install FIRST. The ChimeraX scripts import the INSTALLED bundle, so without
# this they happily test a stale copy: an edit that deleted three functions
# from render.py passed the whole suite because the installed bundle still
# had them.
echo "=== installing the bundle under test ==="
bash "$ROOT/tools/install_bundle.sh" >/dev/null 2>&1 || {
  echo "   install FAILED"; exit 1; }
INST="$HOME/Library/Application Support/ChimeraX/1.12/lib/python/site-packages/chimerax/hitpickingtool"
if [ -d "$INST" ]; then
  for f in render.py cmd.py viewdock_link.py; do
    if ! diff -q "$ROOT/chimerax/src/$f" "$INST/$f" >/dev/null 2>&1; then
      echo "   WARNING: installed $f differs from the source tree"
      fail=1
    fi
  done
fi

echo "=== core unit tests ==="
"$PY" -m pytest "$ROOT/core/tests" -q || fail=1

for t in tests_viewdock_contract tests_labels tests_json_returns tests_clash_parity tests_clash_draw tests_hover tests_formal_charge tests_torsion_click tests_learn tests_learn_panel; do
  echo
  echo "=== chimerax/$t.py ==="
  out="$("$CX" --nogui --silent --exit --script "$ROOT/chimerax/$t.py" 2>&1)"
  echo "$out" | tail -4
  echo "$out" | grep -qE "OK$" || fail=1
done

echo
[ $fail -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $fail

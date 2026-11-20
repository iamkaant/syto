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

for t in tests_viewdock_contract tests_labels tests_json_returns tests_clash_parity tests_clash_draw tests_hover tests_formal_charge tests_torsion_click tests_learn tests_background tests_learn_panel tests_predict_columns tests_assume tests_label_layers tests_worklist tests_score_terms tests_panel_controls; do
  echo
  echo "=== chimerax/$t.py ==="
  out="$("$CX" --nogui --silent --exit --script "$ROOT/chimerax/$t.py" 2>&1)"
  echo "$out" | tail -4
  echo "$out" | grep -qE "OK$" || fail=1
done

echo
echo "=== web parity (the same core, in a browser) ==="
wp="$("$PY" "$ROOT/tools/check_web_parity.py" 2>&1)"
echo "$wp" | tail -5
echo "$wp" | grep -qE "WEB PARITY OK|skipping the web parity check" || fail=1

echo
echo "=== the web page, driven in a browser ==="
wv="$("$PY" "$ROOT/tools/check_web_viewer.py" 2>&1)"
echo "$wv" | tail -6
echo "$wv" | grep -qE "WEB VIEWER OK|skipping the web viewer check" || fail=1

echo
echo "=== the grading loop, in a browser ==="
wal="$("$PY" "$ROOT/tools/check_web_al.py" 2>&1)"
echo "$wal" | tail -6
echo "$wal" | grep -qE "WEB AL OK|skipping" || fail=1

echo
echo "=== panel JavaScript (executed, not just parsed) ==="
pjs="$("$PY" "$ROOT/tools/check_panel_js.py" 2>&1)"
echo "$pjs" | tail -4
echo "$pjs" | grep -qE "PANEL JS OK|skipping the panel JS check" || fail=1

echo
[ $fail -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $fail

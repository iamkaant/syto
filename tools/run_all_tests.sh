#!/usr/bin/env bash
# Everything that can be checked without a GUI.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CX="${CHIMERAX_APP:-/Applications/ChimeraX-1.12.app}/Contents/bin/ChimeraX"
PY="${HITPICK_PYTHON:-/Users/ak87/miniforge3/envs/admet_ai/bin/python}"
fail=0

echo "=== core unit tests ==="
"$PY" -m pytest "$ROOT/core/tests" -q || fail=1

for t in tests_viewdock_contract tests_labels tests_json_returns tests_clash_parity tests_clash_draw; do
  echo
  echo "=== chimerax/$t.py ==="
  out="$("$CX" --nogui --silent --exit --script "$ROOT/chimerax/$t.py" 2>&1)"
  echo "$out" | tail -4
  echo "$out" | grep -qE "OK$" || fail=1
done

echo
[ $fail -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $fail

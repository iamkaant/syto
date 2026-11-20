# The web front end

The same `hitpicking` core the ChimeraX bundle runs, in a browser tab,
with Mol* drawing it. Not a port: `web/vendor/hitpicking.zip` is built
from `core/hitpicking`, so an interaction is decided in exactly one place.

Load a receptor and a pose file, get the pose in 3D with the scored
interactions drawn on it, and click any atom for the handbook entry.

## Running it

```
python tools/fetch_web_deps.py     # once: Pyodide + numpy + Mol* into web/vendor
python tools/build_web_core.py     # after any change to core/ or ui/
python tools/serve_web.py          # serves web/ and opens it
```

**It will not work from `file://`.** Module workers and WebAssembly are
both blocked there - a browser rule, not something the page can work
around - and the page says so rather than failing silently. That is the
whole reason `serve_web.py` exists.

**Nothing is uploaded.** The server hands the browser the page; the
receptor and the poses are read by the tab from your disk and analysed in
it. They never cross the network, not even to localhost.

## What is here

| | |
|---|---|
| `index.html` | the page: load, view, switch pose, click an atom |
| `js/viewer.js` | Mol* as a renderer - structures, our lines, atom picking |
| `js/worker.js` | module worker: boots Pyodide, unpacks the core, answers calls |
| `js/engine.py` | the thin Python boundary - loads files, calls the core, returns JSON |
| `ui/panel.html` | the handbook panel, the same file ChimeraX loads |
| `ui/bridge.js` | the call interface both hosts implement |
| `vendor/` | Pyodide 314.0.8 + numpy 2.4.6 + Mol* 5.13.1 + the core zip. Gitignored, ~23 MB |

## Why the pieces are shaped this way

**Mol\* renders; it does not decide.** Its built-in interactions
representation is deliberately unused. If it decided what an H-bond was,
this page and the ChimeraX bundle would be two different tools that share
a logo, and `tools/check_web_parity.py` would have nothing left to
compare. Every line on screen is one the core found, scored and labelled.

**Lines are drawn as distance measurements.** That is a drawing
primitive - two atoms, a dashed coloured line, a label. Which two atoms,
what colour and what the label says are entirely ours.

**Endpoints cross as coordinates, not atom indices.** Mol\* parses the
file with its own reader and numbers atoms its own way; a shared index
would be a silent assumption that two parsers agree. Positions are read
verbatim from the file by both.

The matching is by 0.1 Å bucket and nearest-within-0.05 Å, not by a
rounded coordinate used as a key. Mol\* holds coordinates as float32 and
the core as float64, so the same atom is `135.90499877929688` on one side
and `135.905` on the other - and those round to different strings. The
first version keyed on `toFixed(2)` and lost two endpoints in six.

**One handbook panel, not two.** `ui/panel.html` is the file the ChimeraX
bundle loads, vendored here by the build and embedded in an iframe. It
takes its host and theme from the query string, because a parent page
cannot set a property on a frame before that frame's own script runs.
Two renderers would be two chances to explain an atom differently.

**`engine.py` does as little as possible.** Every line of judgement it
contains is a line that can disagree with ChimeraX. It loads files, calls
`ReceptorContext.analyze`, and serialises the result.

**Everything crosses as JSON strings.** One conversion, no proxies to
leak, and the bytes are directly diffable against the other front end.

**numpy is pinned to 2.4.6**, the version ChimeraX 1.12 ships. Parity is
easier to believe when the arithmetic is the same build.

## How it is checked

`tools/check_web_parity.py` runs the same receptor and pose through both
hosts and compares every interaction term for term, plus the handbook
answer for several atoms.

`tools/check_web_viewer.py` drives this page in a real browser: feeds it
the files, waits for the analysis, and asserts that every endpoint the
core produced was matched to an atom in the viewer, that all the lines
were drawn, that clicking an atom fills the handbook panel, that an atom
outside the analysis is refused rather than guessed at, and that
switching pose redraws instead of accumulating.

Both run in `tools/run_all_tests.sh`.

## Not done yet

- **No active learning.** The core supports it; this page has no grading
  UI. The ChimeraX bundle is where a party is run today.
- No torsion list or strain panel in the web page; the data is there
  (`analyse` returns it) and the 3D view does not yet show it.
- Pyodide boot is ~2 s and Mol* adds 5.3 MB. Acceptable, not optimised.
- The pose list is flat. A few thousand poses want sorting and filtering,
  which is what ViewDock gives the other front end.

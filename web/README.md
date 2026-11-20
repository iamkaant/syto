# The web front end

The same `hitpicking` core the ChimeraX bundle runs, in a browser tab,
with Mol* drawing it. Not a port: `web/vendor/hitpicking.zip` is built
from `core/hitpicking`, so an interaction is decided in exactly one place.

Load a receptor and a pose file, get the pose in 3D with the scored
interactions drawn on it, and click any atom for the handbook entry.

## The three ways to run it, and why there are only three

**A hosted URL is the one to use.** `.github/workflows/pages.yml`
publishes `web/` to GitHub Pages on every push. Open the link, upload a
receptor and some poses, done - no install, no server, nothing to
remember. The hosting serves the page and nothing else: your files are
read and analysed inside the tab and never cross the network.

**That URL then works offline.** `sw.js` caches the whole runtime on the
first visit, so afterwards it opens with no network at all. This is what
"have it locally" means in a browser: the page is local once you have
seen it once.

**Double-clicking `index.html` cannot be made to work.** From a `file://`
origin Chrome refuses both of the things this needs:

```
new Worker("js/worker.js", {type:"module"})
  -> SecurityError: ... cannot be accessed from origin 'null'
fetch("vendor/hitpicking.zip")
  -> TypeError: Failed to fetch
```

That is a browser security rule about origins, not a limitation of this
page, and no page can opt out of it. The only way around it would be to
inline all 22 MB of Pyodide, numpy and Mol* as base64 in one ~30 MB HTML
file *and* monkey-patch Pyodide's internal asset loading - unsupported,
and it would break on the next version bump. So the page detects
`file://` and says so plainly instead of sitting on "starting..." while a
worker fails to construct.

For a copy served from your own machine, `python tools/serve_web.py`
binds loopback and opens it.

## Building it



```
python tools/fetch_web_deps.py     # once: Pyodide + numpy + Mol* into web/vendor
python tools/build_web_core.py     # after any change to core/ or ui/
python tools/serve_web.py          # serves web/ and opens it
```

The Pages workflow runs exactly these three steps, so what is deployed is
what you get locally.

## What is here

| | |
|---|---|
| `index.html` | the page: load, view, switch pose, click an atom |
| `js/viewer.js` | Mol* as a renderer - structures, our lines, atom picking |
| `js/worker.js` | module worker: boots Pyodide, unpacks the core, answers calls |
| `js/engine.py` | the thin Python boundary - loads files, calls the core, returns JSON |
| `ui/panel.html` | the handbook panel, the same file ChimeraX loads |
| `ui/bridge.js` | the call interface both hosts implement |
| `sw.js` | caches the runtime so the hosted page works offline |
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

`tools/check_web_al.py` runs a whole party in a browser: featurise,
grade ten poses across all three grades, grade one more by hotkey, train,
check the suggestions exclude what you already graded, then save the
party and load it back - including a party from a different file, which
must be reported rather than applied to whatever is at that index.

`tools/check_web_viewer.py` drives this page in a real browser: feeds it
the files, waits for the analysis, and asserts that every endpoint the
core produced was matched to an atom in the viewer, that all the lines
were drawn, that clicking an atom fills the handbook panel, that an atom
outside the analysis is refused rather than guessed at, and that
switching pose redraws instead of accumulating.

Both run in `tools/run_all_tests.sh`.

## Running a party

Analyse all poses, grade with **b** / **m** / **g** (ViewDock's keys, so
both front ends are graded the same way), train, and the page hands you
the ones worth looking at next. It is `hitpicking.al` in the tab - the
same committee, the same acquisition, the same features as ChimeraX - so
a party started here can be saved and finished there.

Featurising is the slow step and the one that decides whether this can
host a party rather than only review one. Measured with
`tools/bench_web_al.py` on a real receptor:

| | native Python | Pyodide |
|---|---|---|
| per pose | 0.063 s | 0.144 s |
| 500 poses | 31 s | 72 s |
| 2000 poses | 2.1 min | 4.8 min |

2.3x slower, and 99% of it is the interaction and strain analysis rather
than building the vector - so a pose you have already looked at is
effectively already featurised. It runs in chunks with a progress bar,
and the model trains on whatever is done, so the first useful ranking is
seconds away rather than minutes.

Grades are kept in `localStorage` against the pose file's name, so a
reload does not lose the afternoon. **Save party** writes the same JSON
`hitpick learn save` writes.

## Not done yet

- No torsion list or strain panel in the web page; the data is there
  (`analyse` returns it) and the 3D view does not yet show it.
- No ViewDock-style sortable table. The pose list carries your grade and
  the predicted score, but a few thousand poses want filtering.
- Pyodide boot is ~2 s and Mol* adds 5.3 MB. Acceptable, not optimised.
- The pose list is flat. A few thousand poses want sorting and filtering,
  which is what ViewDock gives the other front end.

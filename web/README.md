# The web front end

The same `hitpicking` core the ChimeraX bundle runs, in a browser tab.
Not a port: `web/vendor/hitpicking.zip` is built from `core/hitpicking`,
so an interaction is decided in exactly one place.

## Running it

```
python tools/fetch_web_deps.py     # once: Pyodide + numpy into web/vendor
python tools/build_web_core.py     # after any change to core/ or ui/
python -m http.server --directory web 8000
```

Then open <http://127.0.0.1:8000/>.

**It will not work from `file://`.** Module workers and WebAssembly are
blocked there, and the page says so rather than failing silently.

## What is here

| | |
|---|---|
| `index.html` | the page: load a receptor and poses, analyse, read the interactions |
| `js/worker.js` | module worker: boots Pyodide, unpacks the core, answers calls |
| `js/engine.py` | the thin Python boundary - loads files, calls `ReceptorContext.analyze`, returns JSON |
| `ui/` | copied from `/ui` by the build; panels shared with ChimeraX |
| `vendor/` | Pyodide 314.0.8 + numpy 2.4.6 + the core zip. Gitignored, ~17 MB |

## Why the pieces are shaped this way

**`engine.py` does as little as possible.** Every line of judgement it
contains is a line that can disagree with ChimeraX. It loads files, calls
the core, and serialises the result.

**Everything crosses as JSON strings.** One conversion, no proxies to
leak, and the bytes are directly diffable against the other front end -
which is what `tools/check_web_parity.py` does on every test run: same
receptor, same pose, both hosts, every interaction compared term for term.

**numpy is pinned to 2.4.6**, the version ChimeraX 1.12 ships. Parity is
easier to believe when the arithmetic is the same build.

## Not done yet

- **No 3D viewer.** Mol* is the plan; the page currently lists
  interactions without drawing them. This is the next piece of work and
  the largest.
- No handbook panel. `engine.explain()` exists and returns the entry plus
  the pKBHX/pKBH+ scales, but nothing calls it: there is no atom to click
  without a viewer.
- No active learning. The core supports it; the page has no grading UI.
- Pyodide boot is ~2 s. Acceptable, not yet optimised.

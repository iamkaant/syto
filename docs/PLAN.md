# HitPickingTool — implementation plan

## Context

Hit-picking parties ([wiki.docking.org](https://wiki.docking.org/index.php?title=Hit_picking_party)) are the human judgement step before compounds get purchased: an investigator picks "a dozen or two interesting compounds from among the top 500", the group looks at one pose at a time, people call out favourable and unfavourable features, and each compound gets 1–3 checkmarks. The wiki's own criteria list is the de-facto spec:

> Think about enthalpic and entropic contributions · Think about possible water structure · Think of receptor flexibility, especially His, Gln, Asn, Ser, Tyr, Thr · Look for charged residues · Look at the surface coloured by charge to look for greasy patches · Watch for weird conformations and weird protonation states and weird tautomeric forms · Watch out for super floppy molecules · **Watch out for buried polarity or missed opportunities**

Today all of that lives in people's heads. A junior person at the party has no way to know that a pyridine N is a far better acceptor than an amide O, or that the H-bond they're admiring is 40° off-axis. Nothing records *why* a compound was rated.

**The goal:** a helper that (1) puts the quantitative interaction literature at the user's fingertips, contextually — click an atom, get the numbers for that moiety with the relevant atoms highlighted; (2) draws and *scores* the non-covalent interactions and flags clashes and strain; (3) learns from the ratings the group gives, via AutoParty-style active learning, so the tool re-ranks the remaining poses as the party proceeds.

Two front ends, one engine: a **ChimeraX 1.12 bundle** that rides alongside ViewDock (built first, proven at a real party), and a **static web page** that needs no install.

---

## Decisions taken

| Question | Decision |
|---|---|
| Build order | **ChimeraX bundle first.** The lab already views DOCK mol2 in ViewDock; real 3D and the rating UI come for free. |
| Web architecture | **Static page + optional sidecar.** Pyodide runs the *same* Python core in-browser, so the static page is fully functional — including AL training. The sidecar is an optional upgrade, auto-detected, installed via a copy-paste one-liner offered by the page itself. |
| AutoParty | **Extract a standalone, numpy-only AL library.** Your fork at `../AutoParty` is left untouched. |
| Reference data | Build from open sources now; a **literature-extraction prompt** (Appendix A) gets the paywalled material in a browser-enabled session. |

### Why numpy instead of PyTorch for the AL core

AutoParty's model is a committee of 3 MLPs — `Linear(~4273, 1024) → ReLU → Dropout → Linear(1024,1024) → ... → Linear(1024, 3) → Sigmoid`, BCE loss, Adam. At the scale of a hit-picking party (hundreds to a few thousand poses) a hand-written numpy forward/backward trains in seconds. Dropping torch is what lets the identical code run in ChimeraX's Python 3.11 (no torch bundled) **and** in Pyodide. A dev-only parity test pins it against the torch original on a fixed seed.

### Why not LUNA

LUNA needs Python 3.7, openbabel and biopython 1.72. It cannot run inside ChimeraX's Python 3.11 or in Pyodide. Our own NCI engine produces the interaction fingerprint instead; an import path accepts a LUNA/AutoParty feature table (`fp`, `label` columns) for users who have one.

---

## Repository layout

```
HitPickingTool/
  core/                      # pip-installable "hitpicking" — pure Python + numpy ONLY
    hitpicking/
      molsys.py              # MolSystem: numpy arrays — xyz, element, idatm-like hyb/charge class,
                             #   aromatic, formal charge, nH, bonds, ring lists, res/chain, is_ligand
      io/                    # mol2 (DOCK/Sybyl), sdf, pdb parsers -> MolSystem
      perceive/
        matcher.py           # restricted-SMARTS subgraph matcher (NO bond-order primitives)
        rings.py, moieties.py
      nci/                   # hbond, saltbridge, pistack, cationpi, halogen, scoring
      strain/                # clash, torsion, geometry
      handbook/              # loader, atom -> entry lookup, per-atom value annotation
      al/                    # features, model (numpy MLP ensemble), acquisition, io
      data/handbook.json     # compiled, versioned, shipped in the wheel
    tests/
  data-src/                  # hand-edited YAML — the curation surface
  tools/                     # build_data.py, rdkit_annotate.py (offline, optional)
  ui/                        # shared JS panels + bridge.js (HPBridge interface)
  web/                       # static page: index.html, vendored Mol*, Pyodide worker
  chimerax/                  # the bundle
  sidecar/                   # optional local server (FastAPI), same core + RDKit
  docs/
```

Critical files to create first: `core/hitpicking/molsys.py`, `core/hitpicking/perceive/matcher.py`, `ui/bridge.js`, `chimerax/src/tool.py`.
Reference for the AL port: `../AutoParty/app/base/models.py` (`BaseModel`, `EnsembleModel`, `FingerprintDataset`) and the training loop in `app/base/tasks.py::task_train_model`.

---

## Module 1 — the handbook

### 1a. Reference data (contextual)

`data-src/*.yaml` is the source of truth; `tools/build_data.py` compiles it to a versioned `handbook.json` with a schema version and content hash. Both front ends read only the JSON.

- **Moiety**: `{id, name, smarts, roles: {acceptor_atom: 0}, tags, ionization_notes}`
- **Entry**: `{id, title, moiety_ids[], interaction_type, values: [{quantity, value|range, unit, conditions, ref}], geometry: {d_min, d_ideal, d_max, angle terms}, energy_kcal: {low, typ, high, tier}, refs: [{doi}], notes, curation_status}`

**Every number carries a DOI and a `curation_status` (`verified` | `needs_review`). Nothing is auto-generated.** `build_data.py` fails the build on a value without a reference. Curating ~100 entries by hand is the real work here, and it needs a lab member's eyes.

The contextual link is `moiety_id`: clicking an atom calls `handbook.lookup(atom_idx)` → moiety hits, entry ids, and the atom indices to highlight. The panel scrolls to `#entry-id`; the front end highlights.

v0.1 coverage: H-bond strength / pKBHX basicity, salt bridges, π-stacking, cation-π, halogen bonds, desolvation penalty — ~30 moieties.

### 1b. Values on atoms in 3D

ChimeraX: `chimerax.label.label3d.label()` on the matched atoms. Web: Mol* labels. Driven by the same `handbook.annotate(molsys)` call returning `{atom_idx: [{quantity, value, unit}]}`. Per-quantity toggle.

### 1c. Detected NCIs, scored

Our own detectors in `core/hitpicking/nci/` — **not** `chimerax.hbonds` — so both front ends produce byte-identical output. `chimerax.hbonds.find_hbonds` and `chimerax.clashes.find_clashes` are used in the *test suite* as a cross-check only.

Scoring: `quality = f(distance) * g(angles)`, a soft score in [0,1] against the handbook ideal; reported strength is `quality` scaled into the handbook's `energy_kcal` range. **Presented as a heuristic ranking aid, never as a free energy, and never summed into a single score.**

Rendering: ChimeraX `structure.pseudobond_group(name, create_type=None)` + `pbg.new_pseudobond()` with per-type colour/dashes; web uses a custom Mol* Shape representation. Per-type toggles and a show-lengths toggle, as requested.

### 1d. Clashes and strain

- **Clash** (core): vdW overlap with an H-bond allowance, spatial hash, inter- and intramolecular (1-5 and beyond).
- **Geometry/eclipsed torsions** (core): sp3–sp3 torsions within ~20° of eclipsed; bond length/angle deviation vs curated hybridisation-based values; sp2 and aromatic planarity.
- **Torsion library** (v0.4): the Roche torsion library (Schärfer 2013, Guba 2016) is openly published, and RDKit's ETKDG torsion preferences are BSD-licensed and CSD-derived — **check each licence before bundling**, convert to our SMARTS subset.
- **CSD/Mogul: be honest — it is licensed, not redistributable, and cannot run in a browser.** Ship an *optional adapter* a licensed user runs offline to produce `mogul_annotations.json` for import.
- **OpenMM + GAFF** (ChimeraX only, v0.4, optional): single-point energy, then restrained minimisation, report ΔE. **Must verify first** whether ChimeraX 1.12's AmberTools (`antechamber`, `parmchk2`) is usable; if not, this feature is dropped, not faked.

---

## Module 2 — active learning (`hitpicking.al`)

A faithful numpy port of AutoParty's design: ordinal thermometer labels, BCE, Adam, committee of N, `predict() -> (mean, variance)`, acquisition by **uncertainty** (ensemble variance) or **disagreement** (ordinal rank error). Bugs fixed on the way: `np.mean(epoch_losses, val_losses)` in the validation branch; the unimplemented `dropout` and `distance` uncertainty methods are left out rather than left as dead config options.

**Features** (versioned — models are not interchangeable across feature versions):
1. Interaction fingerprint from our NCI engine, hashed over (type, ligand moiety id, protein residue class), folded to 1024 bits, counts optionally weighted by `quality`.
2. Morgan-like ligand hash computed on `MolSystem`, folded to 1024 bits.
3. ~15 descriptors: heavy atoms, rotors, formal charge, rings, clash count, strain score, unsatisfied buried polar atoms, plus DOCK score terms when present.

MACCS requires RDKit → available only via the optional annotation JSON or the sidecar.

**ViewDock integration** (`chimerax/src/viewdock_link.py`) — all of it wrapped in `try/except`, pinned to 1.12, degrading to our own standalone table if ViewDock's internals move:
- Find the tool: `session.tools.find_by_class(ViewDockTool)`.
- Labels come from `structure.viewdock_data['rating']` (`bad`/`maybe`/`good`, hotkeys `b`/`m`/`g`); unrated poses are the pool. This maps 1:1 onto AutoParty's 3-level ordinal scale.
- Listen on the Qt signal `tool.struct_table.selection_changed` — the only hook for "current pose changed".
- Write `AL_score`, `AL_unc`, `AL_rank` into `viewdock_data` then `struct_table.add_column(...)` / `update_column(col, data=True)` — exactly the pattern ViewDock's own `process_hbonds` uses.
- "Next batch" selects top-k by the chosen acquisition. A global off switch disables the module entirely.

---

## Front ends

### ChimeraX bundle

`pyproject.toml` with `[chimerax]`, `[chimerax.tool.*]`, `[chimerax.command.*]`; `class _API(BundleAPI): api_version = 1` with `start_tool` / `register_command` / `get_class`; ends with `bundle_api = _API()`. Build with `devel build`/`devel install`. Docs: `/Applications/ChimeraX-1.12.app/Contents/share/docs/devel/_sources/`.

**Handbook panel: HtmlToolInstance, not native Qt.** Tables, search, collapsible references and anchor-scrolling are far cheaper in HTML, and it means `ui/` is written once for both front ends. ViewDock's own table and the mouse mode stay native.

`ui/bridge.js` defines `HPBridge.call(method, args) -> Promise`, `.on(event, cb)`, `.getSettings()`. ChimeraX implements it over the `hpx:` custom scheme (`handle_scheme`) plus `html_view.runJavaScript` for the return path, with correlation ids. **Day-one spike:** the custom scheme is fire-and-forget and URL-length-limited; if the round trip is painful, switch to `QWebChannel` on `html_view.page()` (PyQt6 6.10.2 supports it). Web implements the same interface over `postMessage` to the Pyodide worker.

Clicking an atom: a `MouseMode` subclass, `event.picked_object(view)` → `PickedAtom.atom` → `handbook.lookup` → panel opens, moiety highlighted.

### Web page

Vendored Pyodide + numpy/scipy wheels and Mol* — offline, ~35–45 MB, no CDN. Core runs in a Web Worker. **Spike:** whether the worker + Pyodide loads from `file://`; if not, ship a one-line `python -m http.server` launcher.

Mol* is used **only** as a renderer. Its built-in interactions representation is deliberately not used: all detections and scores come from the shared core, so the ChimeraX and web outputs stay identical and diffable. Custom scored interaction lines are drawn through a Mol* `Shape`/`Mesh` representation provider (more involved than 3Dmol's `addCylinder` — **spike this alongside the bridge**). Mol* also gives proper molecular-surface charge colouring, which the wiki's "greasy patches" criterion needs.

The **Advanced features** panel probes `localhost` for the sidecar, shows a copy-paste install one-liner if absent, and upgrades live when it answers. Sidecar = optional only: direct disk I/O, RDKit enrichment, LUNA/AutoParty feature import, licensed Mogul data.

---

## Milestones

- **v0.1 — vertical slice.** `MolSystem`, matcher, ~30 moieties, handbook data for 5–6 interaction types, those detectors + per-type and show-length toggles, ChimeraX bundle (mouse mode → handbook + highlight, `hitpick` commands, pseudobond rendering, one ViewDock column), web page with the same panel.
- **v0.2** — per-atom value annotation (pKBHX on a pyridine N), geometric clash + strain.
- **v0.3** — AL module, ViewDock + web integration.
- **v0.4** — torsion library, optional OpenMM strain, Mogul import adapter.
- **v0.5** — sidecar, session save/restore, docs.

### Riskiest unknowns, in order
1. The JS↔Python bridge in `HtmlToolInstance` and a stable highlight/selection round trip. **Spike day one.**
2. Pyodide offline packaging and `file://` loading.
3. ViewDock's internals are not a public API. Wrap defensively; degrade gracefully.
4. **Handbook data quality** — needs human curation and review, not auto-fill.
5. Whether the NCI scoring agrees with what experienced pickers call good. Validate on known complexes.
6. Whether our features match LUNA's informativeness. Compare on a held-out set from `../AutoParty/testing_inputs/`.
7. Protonation/tautomer states — pKBHX and salt-bridge relevance both depend on them. Docking ligands carry explicit H; the protein needs `addh`; the web page must require H in its input.
8. ChimeraX's GAFF toolchain availability.

---

## Verification

- `pytest core/tests` — golden-file tests for parsers, matcher, NCI geometry; **RDKit parity test** (dev-only dep) comparing the matcher against RDKit SMARTS on a few hundred ligands; **ChimeraX parity test** comparing our H-bonds/clashes against `chimerax.hbonds.find_hbonds` and `chimerax.clashes.find_clashes`; **AL parity test** against AutoParty's torch model on a fixed seed.
- `python tools/build_data.py --check` — schema validation, every value has a DOI, no `needs_review` entries in a release build.
- ChimeraX end-to-end: `devel install chimerax/`, then open `../AutoParty/testing_inputs/testset_docked_best.sdf` with `D4.pdb`, confirm ViewDock launches, a click on a pyridine N opens the right handbook entry with the N highlighted, H-bonds are drawn with scores, and toggles take effect live.
- Web end-to-end: open `web/index.html` offline, load the same two files, confirm identical interaction list to the ChimeraX run (a `--dump-json` CLI on the core gives the diff).
- AL end-to-end: rate ~30 poses `g`/`m`/`b` in ViewDock, train, confirm `AL_score`/`AL_unc` columns populate and "next batch" reorders sensibly.
- Real-world: run one actual hit-picking party on it and collect complaints.

---

## Appendix A — literature extraction prompt

Run this in a browser-enabled session (cowork) with journal access. It produces YAML that drops straight into `data-src/`.

> You are building a curated reference dataset of non-covalent interaction parameters for a structure-based drug design tool. Access the sources below, extract the quantitative data, and emit YAML in the schema given at the end. **Accuracy over coverage: never infer, interpolate or invent a number. If a value is not stated in the source, omit the field. Every value must carry the DOI it came from and a page/table/figure locator.**
>
> **Priority sources**
> 1. Laurence et al., "The pKBHX Database: Toward a Better Understanding of Hydrogen-Bond Basicity for Medicinal Chemists", *J. Med. Chem.* 2009, 52, 4073–4086, DOI 10.1021/jm801331y — **especially the Supporting Information**. Extract the full pKBHX scale: for each acceptor family (pyridines, amines, amides, sulfoxides, phosphine oxides, ethers, esters, ketones, nitriles, halogens, aromatics…) the pKBHX value, the specific acceptor atom, substituent effects, and the reference compound. This is the single most valuable table — get the per-family ranges *and* representative individual compounds.
> 2. Bissantz, Kuhn, Stahl, "A Medicinal Chemist's Guide to Molecular Interactions", *J. Med. Chem.* 2010, 53, 5061–5084, DOI 10.1021/jm100112j — the hit-picking wiki cites this directly. Extract every geometric criterion and energy estimate: H-bond distance/angle distributions per donor-acceptor pair, halogen bond σ-hole geometry, π-stacking (parallel-displaced vs T-shaped offsets and energies), cation-π, amide stacking, sulfur interactions, water-mediated bridges, desolvation penalties, and the buried-unsatisfied-polar-atom cost.
> 3. Kuhn et al. on hydration sites / water networks; Schärfer et al. *J. Med. Chem.* 2013 and Guba et al. *J. Chem. Inf. Model.* 2016 on the **Roche torsion library** (extract SMARTS patterns + preferred/tolerated/forbidden torsion ranges — note the licence).
> 4. Allen et al. CSD-derived bond length/angle tables (Allen 1987 and successors) for generic geometry baselines.
> 5. Any open review with tabulated halogen-bond, chalcogen-bond, cation-π or stacking energies.
>
> **For each extracted entry record:** the moiety it applies to (name it precisely — "pyridine N", "primary aliphatic amine N", "aryl amide carbonyl O"), the quantity and unit, the value or range, the measurement conditions (solvent, method — CCl4 vs gas phase vs crystal statistics matters enormously), the DOI, and the exact locator (e.g. "Table 3, row 12" or "SI Table S2").
>
> **Flag explicitly:** values that are gas-phase-only; values from crystal-structure statistics rather than measurement; any value you are less than fully confident you read correctly. Mark all of these `curation_status: needs_review`.
>
> **Emit YAML in this schema:**
> ```yaml
> entries:
>   - id: hbond_pyridine_n
>     title: Pyridine nitrogen (H-bond acceptor)
>     moiety_ids: [pyridine_n]
>     interaction_type: hbond
>     values:
>       - quantity: pKBHX
>         value: 1.86
>         unit: dimensionless
>         conditions: "4-fluorophenol, CCl4, 298 K"
>         ref: {doi: "10.1021/jm801331y", locator: "Table 2"}
>     geometry: {d_ideal: 2.9, d_min: 2.6, d_max: 3.3, angle_dha_min: 120}
>     energy_kcal: {low: -1.0, typ: -2.5, high: -4.5, tier: estimate}
>     notes: "Substituent effects: 4-NMe2 raises to ~2.8; 2-substitution sterically attenuated."
>     curation_status: needs_review
> ```
>
> Also return a plain-text **gaps list**: what you could not access, what was paywalled, and which families have no reliable numbers. Do not fill gaps with guesses.

**If journal access fails**, the fallback is a `papers/` folder in the repo — drop the PDFs in and the same extraction runs locally with the PDF tools, with no licensing ambiguity about redistribution (the compiled `handbook.json` contains only facts and citations, never reproduced text or figures).

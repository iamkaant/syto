# HitPickingTool

A helper for [hit-picking parties](https://wiki.docking.org/index.php?title=Hit_picking_party):
it detects and **scores** the non-covalent interactions in a docked pose, and
puts the quantitative interaction literature one click away from the atom
you are looking at.

Two front ends, one engine:

| | status |
|---|---|
| **ChimeraX bundle**, alongside ViewDock | working (v0.1) |
| **Static web page** (Mol\* + Pyodide) | not started |

The engine (`core/hitpicking`) is pure Python + numpy — no RDKit, no ChimeraX,
no torch — so the identical code runs inside ChimeraX's Python 3.11 and, later,
inside Pyodide in a browser. Both front ends therefore give the same answers by
construction, and a test proves the ChimeraX path and the standalone path agree
atom for atom.

---

## What it does today

**Detects and scores**, against published geometric criteria: hydrogen bonds
(including charge-assisted and water-mediated), salt bridges, parallel and
T-shaped π-stacking, cation–π, halogen bonds, hydrophobic contacts.

Each interaction reports the measured distance and angles, a 0–1 geometric
quality, and plain-language flags such as *"D-H···A 128°, well off linear"* or
*"near-eclipsed sandwich; rings prefer ~1.5 Å offset"*.

**Explains what you click.** Click an atom and you get its chemical role, the
whole moiety highlighted, and the handbook entries for it — including real
pKBHX basicity data from Laurence 2009, every value traceable to a DOI *and* a
locator naming the table it was read from.

**Says what is not an interaction.** Amide and sulfonamide nitrogens, pyrrole
NH and organic fluorine are modelled as explicit negative entries, because half
of what gets argued about at a party is someone over-counting one of these.

### Deliberate limits

- **Scores are heuristic rankings, not free energies.** They are never summed
  into a single number. Energy ranges are coarse, flagged, and shown as ranges.
- **Atom labels show a family median and range** (`pKBHX ~1.9 (0.7–2.7)`), not
  a value for your exact compound: perception knows "a tertiary amine", not
  which substitution pattern.
- **Protonation is treated as evidence, not assumption.** A delocalised cation
  is only called charged when every nitrogen in the group actually bears a
  proton. With no hydrogens in the file, groups are assumed protonated and
  say so.
- **Disagreeing sources stay visible.** 25 values conflict between or within
  sources; they are surfaced, not silently resolved.

---

## Install (ChimeraX)

```bash
tools/install_bundle.sh [/path/to/ChimeraX.app]      # default: ChimeraX-1.12
```

This builds the handbook data, vendors the core into the bundle, builds the
wheel and force-installs it. (Plain `devel install` silently does nothing when
the version has not changed — that is why this script exists.)

Requires PyYAML to build the handbook. If ChimeraX's Python lacks it:

```bash
HITPICK_PYTHON=/path/to/python/with/pyyaml tools/install_bundle.sh
```

## Use

```
open receptor.pdb
open poses.mol2                  # ViewDock opens automatically

hitpick setup                    # perceive the receptor once
hitpick analyze                  # analyse the displayed pose and draw it
hitpick report                   # text listing with geometry and flags

hitpick hide hydrophobic weak_hbond
hitpick show all
hitpick lengths false
hitpick labels pKBHX             # put basicity values on ligand atoms
hitpick labels off
hitpick explain                  # handbook entry for the selected atom
hitpick clear
```

Open **Tools → Binding Analysis → Hit Picking Handbook** for the panel with
per-type checkboxes and the clicked-atom entry.

For click-to-explain in the 3D view:

```
mousemode right "hit picking handbook"
```

With ViewDock running, poses are analysed as you step through the table, and
four columns are added (`HP HBonds`, `HP Salt`, `HP Stack`, `HP Score`).

---

## Development

```bash
pip install -e core[dev]
pytest core/tests -q                  # 49 tests
python tools/build_data.py            # compile data-src -> handbook.json
python tools/build_data.py --check    # validate without writing
```

The build **fails** on any numeric value lacking a DOI and a locator. That is
the project's core promise, and a test enforces it across the whole dataset.

Perception is cross-checked against RDKit on 1091 real docked ligands
(54,264 atoms): aromaticity 0.031%, hybridisation 0.006%, ring count 0.017%
disagreement. RDKit is a dev-only dependency; nothing in the package imports it.

### Layout

```
core/hitpicking/     the shared engine (numpy only)
  molsys.py          atoms/bonds, spatial hash, rings, geometry
  io/                mol2 (DOCK score block), SDF, PDB readers
  perceive/          donors, acceptors, charged groups, conjugation
  nci/               criteria (cited) + detection and scoring
  handbook/          contextual lookup; click -> entry + highlight
  analysis.py        ReceptorContext: perceive once, analyse many poses
data-src/            handbook sources; external/ holds extracted literature
chimerax/            the bundle (adapter.py is the only ChimeraX-aware file)
tools/               build_data.py, vendor_core.py, install_bundle.sh
```

---

## Data provenance

Geometric criteria: Baker & Hubbard 1984, Mills & Dean 1996, Barlow & Thornton
1983, McGaughey 1998, Gallivan & Dougherty 1999, Auffinger 2004, Bissantz,
Kuhn & Stahl 2010.

Quantitative values: Laurence et al. 2009 (pKBHX, incl. the full SI),
Bissantz 2010, Dougherty 2013, Cavallo 2016, Green & Popelier 2014.
See `data-src/external/GAPS_AND_CONFLICTS.txt` for what is missing and which
values disagree.

**Known gaps:** the Roche torsion library is not usable (the Hamburg dataset
page states "All rights reserved"; the 2016 CSD version needs a CCDC licence),
so torsion-strain detection will rely on geometry plus optional OpenMM rather
than CSD-derived preferences. Allen 1987 bond-length tables were not accessed,
so there are no generic bond-geometry baselines yet.

Nothing in this repository redistributes copyrighted text or figures: the
compiled handbook holds facts and citations only.

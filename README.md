# Syto

Syto (Ukrainian "сито": fine sieve) is a helper for hit-picking:
it detects and **scores** the non-covalent interactions in a docked pose, and
puts the quantitative interaction literature one click away from the atom
you are looking at.

Two front ends, one engine:


 -  **ChimeraX bundle** (`ChimeraX-Syto`), alongside ViewDock 
 -  **Static web page** (Mol\* + Pyodide) 

The engine (`core/hitpicking`) is pure Python + numpy — no RDKit, no ChimeraX,
no torch — so the identical code runs inside ChimeraX's Python 3.11 and, later,
inside Pyodide in a browser. Both front ends therefore give the same answers by
construction, and a test proves the ChimeraX path and the standalone path agree
atom for atom.

---

## What it does 

**Detects and scores**, against published geometric criteria: hydrogen bonds
(including charge-assisted and water-mediated), salt bridges, parallel and
T-shaped π-stacking, cation–π, halogen bonds, hydrophobic contacts.

**Flags clashes and strain**: van der Waals overlaps with the site and inside
the ligand (drawn in red, severe ones separately), puckered aromatic rings,
and the docking program's own strain terms.

**Measures every rotatable torsion**:
an sp3–sp3 bond on an eclipsed rotamer, an aromatic amine or anilide twisted
out of the ring plane with no ortho substituent to force it, a twisted amide,
an ortho-substituted biaryl held unnaturally flat. 

Each interaction reports the measured distance and angles, a 0–1 geometric
quality, and plain-language flags such as *"D-H···A 128°, well off linear"* or
*"near-eclipsed sandwich; rings prefer ~1.5 Å offset"*.

**Explains what you click.** Click an atom and you get its chemical role, the
whole moiety highlighted, and the handbook entries for it — including real
pKBHX basicity data from Laurence 2009, every value traceable to a DOI *and* a
locator naming the table it was read from.

**Predicts grade of unseen ligands via active learning.** Rank a few poses and 
train an ensemble model to predict grades of unseen ligands. Look through the 
more ligands and repeat training. Use the trained model to do automatic 
hit picking of thousands of ligands, and only spend time on grading the best
ones. This feature is a port of [AutoParty](https://github.com/keiserlab/autoparty).

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
The tool is planned to be published on the ChimeraX Toolshed


## Use

```
open receptor.pdb
open poses.mol2                  # ViewDock opens automatically

hitpick setup                    # perceive the receptor once
hitpick analyze                  # analyse the displayed pose and draw it
hitpick report                   # text listing with geometry and flags

hitpick clashes true             # draw van der Waals clashes (red)
hitpick torsions                 # flagged torsions; 'true' for all of them
hitpick limits true              # put the literature limits in the labels
hitpick width 0.05               # thinner lines (ChimeraX's own default is 0.075)
hitpick hover true               # detail in the hover balloon
hitpick panel                    # detach the panel; widen it for two columns
hitpick hide hydrophobic weak_hbond
hitpick show all
hitpick lengths false
hitpick labels pKBHX             # put basicity values on ligand atoms
hitpick labels off
hitpick explain                  # handbook entry for the selected atom
hitpick clear
```

Open **Tools → Binding Analysis → Hit Picking Handbook** for the panel with
per-type checkboxes and the clicked-atom entry. Drag it out of the ChimeraX
window (or run `hitpick panel`) and widen it: past ~680 px it lays out in two
columns, past ~1100 px in three.

**Hovering** replaces ChimeraX's balloon with the handbook's answer: an atom
gives its role and basicity, an interaction line gives every measured term
against its limits, a clash gives its overlap, a rotatable bond gives its
torsion. `hitpick hover false` gives ChimeraX's own balloon back.

**Click-to-explain works out of the box**: select a single atom (ctrl-click)
and its handbook entry appears in the log and the panel. Turn it off with
`hitpick autoexplain false`.

If you would rather have a dedicated mouse button than use selection:

```
mousemode right "hit picking handbook"
```

With ViewDock running, poses are analysed as you step through the table, and
five columns are added: `HP H-Bonds`, `HP Salt`, `HP Stack`, `HP Good` and
`HP Clash`.

`HP Good` counts the interactions whose geometry is comfortably inside its
literature limits (quality >= 0.55); `HP Clash` is the worst van der Waals
overlap in Angstrom. **There is deliberately no single "score" column.** An
earlier one summed every interaction's geometric quality, and on a 150-pose
sample that correlated +0.84 with the mere number of interactions, -0.17
with DOCK's own energy, and left 64 poses carrying a clash over 0.6 A in its
top half. Sort on `HP Good` by all means, but read it next to `HP Clash` and
the pose itself.

---

## Active learning (AutoParty)

Optional and lazily loaded: nothing here is imported until you run one of
these, so the handbook costs nothing if you never use it.

```
hitpick learn collect            # analyse and featurise the poses
                                 # ...then grade in ViewDock with g / m / b
hitpick learn train
hitpick learn next               # the poses worth grading now
hitpick learn next strategy disagreement   # where it disagrees with YOU
hitpick learn status
hitpick learn save party.json
```

There is a panel too: **Tools -> Binding Analysis -> Hit Picking Learn**, or
`hitpick learn panel`. It is a separate tool from the handbook, so the whole
active-learning stack loads when you open it and not before. It gives large
bad / maybe / good buttons for the displayed pose, a ranked list of what to
look at next (click a row to jump to that pose), the training curve, and
save/load. By default grading moves straight on to the next suggestion,
which is the loop a party actually runs.

Grades come from ViewDock's own Rating column, so there is no second place
to record an opinion; `hitpick learn grade good` works without ViewDock.

A numpy port of the [AutoParty](https://github.com/keiserlab/autoparty) model
with PyTorch removed, so it runs inside ChimeraX and, later, in the browser.
A committee of small networks is trained on the ordinal grades, and the poses
it disagrees with itself about are the ones worth grading next.

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
tools/               build_data.py, vendor_core.py, install_bundle.sh,
                     release_bundle.py (the Toolshed wheel)
```

---

## Data provenance

Every number in the handbook carries the DOI it came from and a locator
naming the table or figure it was read from. The sources:

**Geometric criteria**

1. Baker, E. N.; Hubbard, R. E. Hydrogen Bonding in Globular Proteins.
   *Prog. Biophys. Mol. Biol.* **1984**, *44* (2), 97–179.
   [DOI: 10.1016/0079-6107(84)90007-5](https://doi.org/10.1016/0079-6107(84)90007-5)
2. Mills, J. E. J.; Dean, P. M. Three-Dimensional Hydrogen-Bond Geometry and
   Probability Information from a Crystal Survey. *J. Comput.-Aided Mol. Des.*
   **1996**, *10* (6), 607–622.
   [DOI: 10.1007/BF00134183](https://doi.org/10.1007/BF00134183)
3. Barlow, D. J.; Thornton, J. M. Ion-Pairs in Proteins. *J. Mol. Biol.*
   **1983**, *168* (4), 867–885.
   [DOI: 10.1016/S0022-2836(83)80079-5](https://doi.org/10.1016/S0022-2836(83)80079-5)
4. McGaughey, G. B.; Gagné, M.; Rappé, A. K. π-Stacking Interactions:
   Alive and Well in Proteins. *J. Biol. Chem.* **1998**, *273* (25),
   15458–15463.
   [DOI: 10.1074/jbc.273.25.15458](https://doi.org/10.1074/jbc.273.25.15458)
5. Gallivan, J. P.; Dougherty, D. A. Cation-π Interactions in Structural
   Biology. *Proc. Natl. Acad. Sci. U.S.A.* **1999**, *96* (17), 9459–9464.
   [DOI: 10.1073/pnas.96.17.9459](https://doi.org/10.1073/pnas.96.17.9459)
6. Auffinger, P.; Hays, F. A.; Westhof, E.; Ho, P. S. Halogen Bonds in
   Biological Molecules. *Proc. Natl. Acad. Sci. U.S.A.* **2004**, *101* (48),
   16789–16794.
   [DOI: 10.1073/pnas.0407607101](https://doi.org/10.1073/pnas.0407607101)
7. Bissantz, C.; Kuhn, B.; Stahl, M. A Medicinal Chemist's Guide to Molecular
   Interactions. *J. Med. Chem.* **2010**, *53* (14), 5061–5084.
   [DOI: 10.1021/jm100112j](https://doi.org/10.1021/jm100112j)

**Quantitative values**

8. Laurence, C.; Brameld, K. A.; Graton, J.; Le Questel, J.-Y.; Renault, E.
   The p*K*<sub>BHX</sub> Database: Toward a Better Understanding of
   Hydrogen-Bond Basicity for Medicinal Chemists. *J. Med. Chem.* **2009**,
   *52* (14), 4073–4086.
   [DOI: 10.1021/jm801331y](https://doi.org/10.1021/jm801331y)
   (including the full Supporting Information)
9. Dougherty, D. A. The Cation−π Interaction. *Acc. Chem. Res.* **2013**,
   *46* (4), 885–893.
   [DOI: 10.1021/ar300265y](https://doi.org/10.1021/ar300265y)
10. Cavallo, G.; Metrangolo, P.; Milani, R.; Pilati, T.; Priimagi, A.;
    Resnati, G.; Terraneo, G. The Halogen Bond. *Chem. Rev.* **2016**,
    *116* (4), 2478–2601.
    [DOI: 10.1021/acs.chemrev.5b00484](https://doi.org/10.1021/acs.chemrev.5b00484)
11. Green, A. J.; Popelier, P. L. A. Theoretical Prediction of Hydrogen-Bond
    Basicity p*K*<sub>BHX</sub> Using Quantum Chemical Topology Descriptors.
    *J. Chem. Inf. Model.* **2014**, *54* (2), 553–561.
    [DOI: 10.1021/ci400657c](https://doi.org/10.1021/ci400657c)

Bissantz et al. (ref 7) is also a source of quantitative values.

**Torsion energy profiles** are relaxed gas-phase QM torsion drives from the
Open Force Field datasets on the MolSSI QCArchive (OpenFF Group1 Torsions,
Substituted Phenyl Set 1 and v2.0, WBO Conjugated Series v1.0), used under
[CC-BY-4.0](https://github.com/openforcefield/qca-dataset-submission/blob/master/LICENSE).
Energies were converted to kcal/mol relative to each scan's own minimum; none
were recomputed.

**Active learning** is a port of
[AutoParty](https://github.com/keiserlab/autoparty).

See `data-src/external/GAPS_AND_CONFLICTS.txt` for what is missing and which
values disagree.


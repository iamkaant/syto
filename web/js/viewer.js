// The 3D view: Mol* as a renderer, and nothing more.
//
// Mol* parses the files and draws atoms; every line between them comes
// from our own core, through engine.scene(). Mol*'s built-in interactions
// representation is deliberately not used - if it decided what an H-bond
// was, the web page and the ChimeraX bundle would be two different tools
// that happen to share a logo, and tools/check_web_parity.py would have
// nothing left to compare.
//
// Lines are drawn as distance measurements. That is a drawing primitive:
// it takes two atoms and puts a dashed, coloured, labelled line between
// them. Which two atoms, what colour and what the label says are entirely
// ours.

(function (global) {
  "use strict";

  // One colour per interaction type, matching the ChimeraX bundle's
  // pseudobond colours so a pose looks the same in both.
  var COLORS = {
    hbond: 0x4b9fff,
    hbond_charged: 0x7ad1ff,
    hbond_weak: 0x6a7f99,
    salt_bridge: 0xff7ac2,
    pi_stack_parallel: 0xb98cff,
    pi_stack_tshaped: 0xb98cff,
    cation_pi: 0xffc24b,
    halogen_bond: 0x4bd4a8,
    hydrophobic: 0x8b93a3
  };
  var CLASH = 0xe64646;
  var SEVERE = 0xff2020;
  var TORSION = 0xffb020;  // the picked torsion, a colour nothing else uses
  // The canvas is white. A light colour that reads well as a 3 pixel
  // line is almost invisible as text on it, so text gets a darker shade
  // of its line's colour and the residue names get near-black.
  var SITE_LABEL = 0x2a2f3a;

  function darken(color, f) {
    var r = Math.round(((color >> 16) & 255) * f);
    var g = Math.round(((color >> 8) & 255) * f);
    var b = Math.round((color & 255) * f);
    return (r << 16) | (g << 8) | b;
  }

  // Thin enough to see the ligand through. 0.22 drew the pocket as
  // tubing and buried the pose inside it.
  var SITE_SIZE = 0.1;

  // Mol* sizes label text in world units, so a label legible on a small
  // ligand is unreadable across a whole pocket and vice versa. There is
  // no single right value; it is a control.
  var TEXT_SIZE = 0.33;

  function Viewer(elementId) {
    this.id = elementId;
    this.siteSize = SITE_SIZE;
    this.siteLabels = false;   // residue names on the pocket, off by default
    this.lineSize = 1.0;        // multiplier on the interaction lines
    this.textSize = TEXT_SIZE;  // the label beside each line
    this.plugin = null;
    this.receptor = null;     // the state cell
    this.trajectory = null;
    this.pose = null;
    this.poseModel = null;    // the parent of this.pose; deleting it
                              // takes the structure with it
    this.measurements = [];
    this.torsion = null;      // the highlighted torsion, if any
    this._index = null;       // rounded coordinate -> {unit, element}
    this.onAtomClick = null;
    this.onBondClick = null;   // a bond is a torsion question, not an atom one
    this.onEmptyClick = null;  // clicking nothing is how you clear a pick
    this.onOtherClick = null;  // anything else, so it can be reported
    this.lastLociKind = null;
  }

  Viewer.prototype.start = async function () {
    var v = await global.molstar.Viewer.create(this.id, {
      layoutIsExpanded: false,
      layoutShowControls: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      layoutShowRemoteState: false,
      viewportShowExpand: true,
      viewportShowSelectionMode: false,
      viewportShowAnimation: false,
      pdbProvider: "rcsb",
      emdbProvider: "rcsb"
    });
    this.viewer = v;
    this.plugin = v.plugin;
    this._wireClick();
    return this;
  };

  // --------------------------------------------------------- structures

  Viewer.prototype.loadReceptor = async function (text, format) {
    var p = this.plugin;
    var data = await p.builders.data.rawData({data: text, label: "receptor"});
    var traj = await p.builders.structure.parseTrajectory(data, format || "pdb");
    var model = await p.builders.structure.createModel(traj);
    var struct = await p.builders.structure.createStructure(model);
    await p.builders.structure.representation.applyPreset(
      struct, "polymer-and-ligand");
    this.receptor = struct;
    this._index = null;
    return struct.data.elementCount;
  };

  Viewer.prototype.loadPoses = async function (text, format) {
    var p = this.plugin;
    var data = await p.builders.data.rawData({data: text, label: "poses"});
    this.trajectory = await p.builders.structure.parseTrajectory(
      data, format || "mol2");
    return this.trajectory.data.frameCount;
  };

  // Everything that builds state runs through here, one at a time.
  //
  // Building a pose is four awaits long. Two of them overlapping - which
  // is what holding down "next" does - means both read this.pose before
  // either writes it, so each deletes a node the other had already
  // replaced and both leave their own structure behind: several ligands
  // on screen at once. A queue is the fix; a flag would only drop the
  // click that happens to arrive mid-build.
  Viewer.prototype._serial = function (fn) {
    var prev = this._chain || Promise.resolve();
    var next = prev.then(fn, fn);
    // the chain must survive a failure, or one error stops every later
    // pose from ever being built
    this._chain = next.catch(function () {});
    return next;
  };

  // One trajectory, one model per pose: switching is a model index, not a
  // re-parse of a file that may hold a few thousand records.
  Viewer.prototype.showPose = function (i) {
    var self = this;
    return this._serial(function () { return self._showPose(i); });
  };

  Viewer.prototype._showPose = async function (i) {
    var p = this.plugin;
    // Delete the MODEL, not the structure built from it.
    //
    // createModel adds a model cell and createStructure hangs a
    // structure off it; deleting the structure leaves the model behind,
    // so every pose switch left one more orphan in the state tree. They
    // draw nothing, which is why this went unnoticed - but they are the
    // parsed coordinates of every pose ever looked at. Deleting the
    // parent takes its subtree with it.
    if (this.poseModel) {
      var old = this.poseModel;
      this.poseModel = null;
      this.pose = null;
      await p.build().delete(old).commit();
    }
    if (!this.trajectory) return 0;
    var model = await p.builders.structure.createModel(
      this.trajectory, {modelIndex: i});
    this.poseModel = model;
    var struct = await p.builders.structure.createStructure(model);
    await p.builders.structure.representation.applyPreset(struct, "auto");
    this.pose = struct;
    this._index = null;
    this.siteLoci = null;
    // The torsion highlight hangs off the pose node, so deleting the pose
    // already removed it. Forgetting the ref keeps a later clear from
    // trying to delete a node that is not there.
    this.torsion = null;
    return struct.data.elementCount;
  };

  // ------------------------------------------------- coordinate -> atom

  // A 0.1 A bucket, NOT a rounded coordinate used as an identity.
  //
  // The first attempt keyed on x.toFixed(2) from both sides and lost two
  // endpoints in six. Mol* holds coordinates as float32 and the core as
  // float64, so the same atom is 135.90499877929688 on one side and
  // 135.905 on the other, and those round to different strings. Rounding
  // can never be made to agree at a boundary; the fix is to stop asking
  // it to. Atoms are never closer than about 0.9 A, so a bucket plus a
  // nearest-within-tolerance search over its neighbours is exact.
  var BUCKET = 10;          // buckets per Angstrom
  var TOL2 = 0.05 * 0.05;   // same atom, in A^2

  function key(x, y, z) {
    return Math.round(x * BUCKET) + "|" + Math.round(y * BUCKET) + "|"
         + Math.round(z * BUCKET);
  }

  // Built once per pose, over whatever structures are loaded. Mol* and
  // the core each number atoms their own way; the position is what they
  // both read verbatim out of the file, so it is what they are matched on.
  Viewer.prototype._buildIndex = function () {
    var SE = global.molstar.lib.structure.StructureElement;
    var SP = global.molstar.lib.structure.StructureProperties;
    var map = {};
    [this.receptor, this.pose].forEach(function (cell) {
      if (!cell || !cell.data) return;
      cell.data.units.forEach(function (unit) {
        var loc = SE.Location.create(cell.data, unit, unit.elements[0]);
        for (var k = 0; k < unit.elements.length; k++) {
          loc.element = unit.elements[k];
          var x = SP.atom.x(loc), y = SP.atom.y(loc), z = SP.atom.z(loc);
          var kk = key(x, y, z);
          (map[kk] || (map[kk] = [])).push(
            {structure: cell.data, unit: unit, element: unit.elements[k],
             offset: k, x: x, y: y, z: z});
        }
      });
    });
    this._index = map;
  };

  Viewer.prototype.atomAt = function (xyz) {
    if (!this._index) this._buildIndex();
    var x = xyz[0], y = xyz[1], z = xyz[2];
    var bx = Math.round(x * BUCKET), by = Math.round(y * BUCKET),
        bz = Math.round(z * BUCKET);
    var best = null, bestD = TOL2;
    for (var i = -1; i <= 1; i++) {
      for (var j = -1; j <= 1; j++) {
        for (var k = -1; k <= 1; k++) {
          var bin = this._index[(bx + i) + "|" + (by + j) + "|" + (bz + k)];
          if (!bin) continue;
          for (var n = 0; n < bin.length; n++) {
            var a = bin[n];
            var dx = a.x - x, dy = a.y - y, dz = a.z - z;
            var d = dx * dx + dy * dy + dz * dz;
            if (d <= bestD) { bestD = d; best = a; }
          }
        }
      }
    }
    return best;
  };

  Viewer.prototype.atomCount = function () {
    if (!this._index) this._buildIndex();
    var n = 0, m = this._index;
    for (var k in m) n += m[k].length;
    return n;
  };

  // A single-atom loci. Mol* indexes a unit's atoms by OFFSET into
  // unit.elements, not by the element id, and its OrderedSet accepts a
  // sorted plain array - so one offset in an array is the whole thing.
  function lociFor(hit) {
    var SE = global.molstar.lib.structure.StructureElement;
    return SE.Loci(hit.structure,
                   [{unit: hit.unit, indices: [hit.offset]}]);
  }

  // A loci over several atoms. Mol* measures between the centres of the
  // two loci it is given, so a whole ring in one loci is a line drawn to
  // that ring's centroid - which is where the core measured from. Drawn
  // from one ring atom instead, a good stack looks like a bad one.
  Viewer.prototype.lociOver = function (coords) {
    var SE = global.molstar.lib.structure.StructureElement;
    var byUnit = new Map();
    var structure = null;
    for (var i = 0; i < coords.length; i++) {
      var hit = this.atomAt(coords[i]);
      if (!hit) continue;
      structure = hit.structure;
      if (!byUnit.has(hit.unit)) byUnit.set(hit.unit, []);
      byUnit.get(hit.unit).push(hit.offset);
    }
    if (!structure) return null;
    var elements = [];
    byUnit.forEach(function (offsets, unit) {
      offsets.sort(function (a, b) { return a - b; });
      elements.push({unit: unit, indices: offsets});
    });
    return SE.Loci(structure, elements);
  };

  // The cartoon is context. It is also most of what you see when the
  // camera is anywhere near the protein, and it hides the pocket sticks
  // inside it, so it starts hidden and is a toggle rather than a given.
  Viewer.prototype.setRibbon = function (on) {
    this.ribbon = !!on;
    var found = false;
    try {
      this.plugin.state.data.cells.forEach(function (cell) {
        var o = cell.obj;
        if (!o || !o.label) return;
        if (o.label !== "Cartoon") return;
        found = true;
      });
      // Visibility is a state flag, not a parameter: toggling it does not
      // rebuild the representation.
      var refs = [];
      this.plugin.state.data.cells.forEach(function (cell, ref) {
        if (cell.obj && cell.obj.label === "Cartoon") refs.push(ref);
      });
      var p = this.plugin;
      refs.forEach(function (ref) {
        p.state.data.updateCellState(ref, {isHidden: !on});
      });
    } catch (e) { /* nothing to hide yet */ }
    return found;
  };

  // ------------------------------------------------------- binding site
  //
  // The residues lining the pocket, as sticks. Which atoms those are is
  // decided by the core (engine.site_atoms) and arrives as coordinates,
  // for the same reason the interaction endpoints do: the viewer's reader
  // numbers atoms its own way and anything decided here could disagree
  // with ChimeraX. The page turns them into whole residues - half an
  // arginine is not a thing anyone wants to look at.

  Viewer.prototype.clearSite = async function () {
    if (!this.site) return;
    try { await this.plugin.build().delete(this.site).commit(); }
    catch (e) { /* already gone */ }
    this.site = null;
  };

  Viewer.prototype.showSite = async function (coords) {
    await this.clearSite();
    if (!coords || !coords.length || !this.receptor) return 0;
    if (!this._index) this._buildIndex();

    var SE = global.molstar.lib.structure.StructureElement;
    var byUnit = new Map();
    var found = 0;
    for (var i = 0; i < coords.length; i++) {
      var hit = this.atomAt(coords[i]);
      if (!hit || hit.structure !== this.receptor.data) continue;
      if (!byUnit.has(hit.unit)) byUnit.set(hit.unit, []);
      byUnit.get(hit.unit).push(hit.offset);
      found++;
    }
    if (!found) return 0;

    var elements = [];
    byUnit.forEach(function (offsets, unit) {
      offsets.sort(function (a, b) { return a - b; });
      elements.push({unit: unit, indices: offsets});
    });
    var loci = SE.Loci(this.receptor.data, elements);
    try { loci = SE.Loci.extendToWholeResidues(loci); } catch (e) { /* as-is */ }
    this.siteLoci = loci;

    var ST = global.molstar.lib.plugin.StateTransforms;
    var bundle = SE.Bundle.fromLoci(loci);
    // The selection is the handle: the sticks hang off it, and so do the
    // residue labels when they are on, so one delete clears both.
    var sel = this.plugin.build().to(this.receptor)
      .apply(ST.Model.StructureSelectionFromBundle,
             {bundle: bundle, label: "binding site"});
    sel.apply(ST.Representation.StructureRepresentation3D, {
      type: {name: "ball-and-stick",
             params: {sizeFactor: this.siteSize, aromaticBonds: false,
                      sizeAspectRatio: 0.5}},
      colorTheme: {name: "element-symbol",
                   params: {carbonColor: {name: "element-symbol",
                                          params: {}}}},
      sizeTheme: {name: "physical", params: {}}
    });
    // One label per residue, not per atom: a pocket labelled atom by atom
    // is unreadable, and "ASP149" is the thing you are trying to read.
    if (this.siteLabels) {
      // residueScale, not textSize: Mol*'s label representation sizes
      // residue-level text with its own scale parameter and ignores a
      // textSize it was never given. The slider's value goes in as-is,
      // which is the same number the interaction labels are drawn at -
      // Mol*'s own default of 1 is three times that and shouts over the
      // pose it is labelling.
      sel.apply(ST.Representation.StructureRepresentation3D, {
        type: {name: "label",
               params: {level: "residue",
                        residueScale: this.textSize,
                        // same reasoning as the interaction labels: the
                        // grey plate and the outline are what make a
                        // small residue name hard to read
                        background: false, backgroundOpacity: 0,
                        borderWidth: 0,
                        attachment: "middle-center"}},
        colorTheme: {name: "uniform", params: {value: SITE_LABEL}},
        sizeTheme: {name: "uniform", params: {}}
      }, {tags: ["hp-sitelabels"]});
    }
    var update = sel;
    await update.commit();
    // Counted so a test can tell a rebuild from a redraw: the pocket
    // blinking on every unrelated redraw is a thing people notice.
    this.siteBuilds = (this.siteBuilds || 0) + 1;
    this.siteCount = found;
    this.site = update.selector ? update.selector.ref : null;
    if (!this.site) {
      // find it back by label so a later clear still works
      this.site = null;
    }
    return found;
  };

  // Frame one interaction: both ends of it, so the line between them is
  // what you are looking at. A row in a list that cannot take you to the
  // thing it describes is a row in a list.
  Viewer.prototype.focusAtoms = function (coords) {
    var loci = this.lociOver(coords || []);
    if (!loci) return 0;
    try {
      this.plugin.managers.camera.focusLoci(loci, {durationMs: 250});
      // and say which one, for as long as the mouse stays away
      this.plugin.managers.interactivity.lociHighlights.highlightOnly(
        {loci: loci});
    } catch (e) { /* no camera yet */ }
    return (coords || []).length;
  };

  // Frame the pocket, not the whole protein. Opening on a 2800-atom
  // cartoon and making the user hunt for the ligand is the single most
  // annoying thing a viewer can do.
  Viewer.prototype.focusSite = function () {
    var loci = this.siteLoci;
    if (!loci && this.pose) {
      loci = global.molstar.lib.structure.Structure.Loci(this.pose.data);
    }
    if (!loci) return;
    try {
      this.plugin.managers.camera.focusLoci(loci, {durationMs: 0});
    } catch (e) { /* no camera yet */ }
  };

  // ---------------------------------------------------------- torsions
  //
  // One torsion, picked from the panel's list, shown as its four atoms in
  // a colour nothing else uses. A list of strained bonds you cannot click
  // through to the geometry is just a table - the whole point is to see
  // which part of the molecule is paying.

  Viewer.prototype.clearTorsion = async function () {
    if (!this.torsion) return;
    try { await this.plugin.build().delete(this.torsion).commit(); }
    catch (e) { /* already gone with the pose */ }
    this.torsion = null;
  };

  // coords are the torsion's atoms, in order, from the core. They are all
  // ligand atoms (the core measures torsions ligand-only), so they belong
  // to the pose structure; anything that does not match it is skipped
  // rather than guessed at.
  Viewer.prototype.showTorsion = async function (coords, focus) {
    await this.clearTorsion();
    if (!coords || !coords.length || !this.pose) return 0;
    if (!this._index) this._buildIndex();

    var SE = global.molstar.lib.structure.StructureElement;
    var byUnit = new Map();
    var found = 0;
    for (var i = 0; i < coords.length; i++) {
      var hit = this.atomAt(coords[i]);
      if (!hit || hit.structure !== this.pose.data) continue;
      if (!byUnit.has(hit.unit)) byUnit.set(hit.unit, []);
      byUnit.get(hit.unit).push(hit.offset);
      found++;
    }
    if (!found) return 0;

    var elements = [];
    byUnit.forEach(function (offsets, unit) {
      offsets.sort(function (a, b) { return a - b; });
      elements.push({unit: unit, indices: offsets});
    });
    var loci = SE.Loci(this.pose.data, elements);

    var ST = global.molstar.lib.plugin.StateTransforms;
    var update = this.plugin.build().to(this.pose)
      .apply(ST.Model.StructureSelectionFromBundle,
             {bundle: SE.Bundle.fromLoci(loci), label: "torsion"})
      .apply(ST.Representation.StructureRepresentation3D, {
        type: {name: "ball-and-stick",
               params: {sizeFactor: 0.45, aromaticBonds: false,
                        sizeAspectRatio: 0.6}},
        colorTheme: {name: "uniform", params: {value: TORSION}},
        sizeTheme: {name: "physical", params: {}}
      }, {tags: ["hp-torsion"]});
    await update.commit();
    this.torsion = update.selector ? update.selector.ref : null;
    if (focus) {
      try { this.plugin.managers.camera.focusLoci(loci, {durationMs: 250}); }
      catch (e) { /* no camera yet */ }
    }
    return found;
  };

  // ------------------------------------------------------------- lines

  // Delete the measurement GROUP rather than remembering each line.
  //
  // addDistance does not hand back anything we can delete later, so the
  // first version collected refs that were always undefined and cleared
  // nothing - every pose would have left its predecessor's lines on
  // screen. The manager keeps all of them under one group node, and that
  // node is a handle.
  Viewer.prototype.clearLines = async function () {
    var p = this.plugin;
    this.measurements = [];
    this.nDrawn = 0;
    var mm = p.managers.structure.measurement;
    var g = null;
    try { g = mm.getGroup(); } catch (e) { g = null; }
    if (g && g.ref) {
      await p.build().delete(g.ref).commit();
    }
  };

  Viewer.prototype.drawScene = async function (scene, opts) {
    opts = opts || {};
    await this.clearLines();
    if (!scene) return 0;
    if (!this._index) this._buildIndex();

    var mm = this.plugin.managers.structure.measurement;
    var self = this;
    var drawn = 0, missed = 0;
    var rows = [];

    (scene.lines || []).forEach(function (l) {
      rows.push({a: l.a, b: l.b,
                 // the ring sides, where the core measured a centroid
                 aSpan: l.a_span && l.a_span.length > 1 ? l.a_span : null,
                 bSpan: l.b_span && l.b_span.length > 1 ? l.b_span : null,
                 color: COLORS[l.kind] === undefined ? 0x8b93a3 : COLORS[l.kind],
                 dash: l.quality >= 0.55 ? 0.1 : 0.45,
                 size: (0.08 + 0.1 * l.quality) * self.lineSize,
                 // No kind name: it was the longest part of every label
                 // and the line's own colour already says it. The legend
                 // in the corner of the view carries the names instead.
                 //
                 // Single spaces. The double ones were set when the kind
                 // name made these labels long, where they read as
                 // grouping; on "2.94 A q 0.80" they are a third of the
                 // string and read as loose letter spacing - which Mol*
                 // gives no way to tighten, because its 3D text has no
                 // such control.
                 text: l.distance.toFixed(2) + " Å q "
                       + l.quality.toFixed(2)});
    });
    if (opts.clashes !== false) {
      (scene.clashes || []).forEach(function (c) {
        rows.push({a: c.a, b: c.b, color: c.severe ? SEVERE : CLASH,
                   dash: 0.2, size: 0.12 * self.lineSize,
                   text: "clash " + c.overlap.toFixed(2) + " Å"});
      });
    }

    this.lastSpans = [];
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      var ha = this.atomAt(r.a), hb = this.atomAt(r.b);
      if (!ha || !hb) { missed++; continue; }
      var la = (r.aSpan && this.lociOver(r.aSpan)) || lociFor(ha);
      var lb = (r.bSpan && this.lociOver(r.bSpan)) || lociFor(hb);
      this.lastSpans.push([r.aSpan ? r.aSpan.length : 1,
                           r.bSpan ? r.bSpan.length : 1]);
      try {
        var cell = await mm.addDistance(la, lb, {
          labelParams: {
            // The outline stays: it is what keeps a number legible when
            // the line behind it runs through dense sticks. Only the
            // plate behind the text is off.
            background: false, backgroundOpacity: 0
          },
          visualParams: {
            linesColor: r.color, linesSize: r.size, dashLength: r.dash,
            textColor: darken(r.color, 0.62), textSize: self.textSize,
            customText: opts.labels === false ? " " : r.text
          }
        });
        if (cell && cell.ref) this.measurements.push(cell.ref);
        drawn++;
      } catch (e) {
        missed++;
      }
    }
    this.lastMissed = missed;
    this.nDrawn = drawn;
    this.lastLabels = rows.map(function (r) { return r.text; });
    return drawn;
  };

  // ------------------------------------------------------------- click

  Viewer.prototype._wireClick = function () {
    var self = this;
    var SP = global.molstar.lib.structure.StructureProperties;
    var SE = global.molstar.lib.structure.StructureElement;
    this.plugin.behaviors.interaction.click.subscribe(function (e) {
      var loci = e.current && e.current.loci;
      if (!loci) return;
      // Kept so a click that goes nowhere can say what it was, instead
      // of looking like a dead control.
      self.lastLociKind = loci.kind;
      if (loci.kind === "empty-loci") {
        if (self.onEmptyClick) self.onEmptyClick();
        return;
      }
      // A bond is its own kind of loci, not two atoms. Clicking the stick
      // between two atoms is how you ask about a torsion - asking about
      // one of its ends instead is a different question.
      if (loci.kind === "bond-loci") {
        if (!self.onBondClick) return;
        try {
          var bd = loci.bonds && loci.bonds[0];
          if (!bd) return;
          var la = SE.Location.create(loci.structure, bd.aUnit,
                                      bd.aUnit.elements[bd.aIndex]);
          var lb = SE.Location.create(loci.structure, bd.bUnit,
                                      bd.bUnit.elements[bd.bIndex]);
          self.onBondClick([SP.atom.x(la), SP.atom.y(la), SP.atom.z(la)],
                           [SP.atom.x(lb), SP.atom.y(lb), SP.atom.z(lb)]);
        } catch (err) { /* a click on nothing is not an error */ }
        return;
      }
      if (loci.kind === "element-loci") {
        if (!self.onAtomClick) return;
        try {
          var l = SE.Loci.getFirstLocation(loci);
          if (!l) return;
          self.onAtomClick([SP.atom.x(l), SP.atom.y(l), SP.atom.z(l)],
                           SP.atom.label_atom_id(l));
        } catch (err) { /* a click on nothing is not an error */ }
        return;
      }
      // Anything else - a whole structure, a shape, a measurement line.
      // Saying so beats silence: silence is indistinguishable from the
      // feature being broken, which is how this was reported.
      if (self.onOtherClick) self.onOtherClick(loci.kind);
    });
  };

  Viewer.prototype.focusPose = function () {
    if (!this.pose) return;
    this.plugin.managers.camera.focusLoci(
      global.molstar.lib.structure.Structure.Loci(this.pose.data));
  };

  // The colours are exported so the legend names exactly what was
  // drawn. A second copy of them in the page would be a legend that
  // can quietly stop matching the picture.
  global.HPViewer = {Viewer: Viewer, COLORS: COLORS,
                     CLASH: CLASH, SEVERE: SEVERE, TORSION: TORSION};
})(window);

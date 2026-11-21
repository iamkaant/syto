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

  // Thin enough to see the ligand through. 0.22 drew the pocket as
  // tubing and buried the pose inside it.
  var SITE_SIZE = 0.1;

  function Viewer(elementId) {
    this.id = elementId;
    this.siteSize = SITE_SIZE;
    this.lineSize = 1.0;        // multiplier on the interaction lines
    this.plugin = null;
    this.receptor = null;     // the state cell
    this.trajectory = null;
    this.pose = null;
    this.measurements = [];
    this._index = null;       // rounded coordinate -> {unit, element}
    this.onAtomClick = null;
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

  // One trajectory, one model per pose: switching is a model index, not a
  // re-parse of a file that may hold a few thousand records.
  Viewer.prototype.showPose = async function (i) {
    var p = this.plugin;
    if (this.pose) {
      await p.build().delete(this.pose).commit();
      this.pose = null;
    }
    if (!this.trajectory) return 0;
    var model = await p.builders.structure.createModel(
      this.trajectory, {modelIndex: i});
    var struct = await p.builders.structure.createStructure(model);
    await p.builders.structure.representation.applyPreset(struct, "auto");
    this.pose = struct;
    this._index = null;
    this.siteLoci = null;
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
    var update = this.plugin.build().to(this.receptor)
      .apply(ST.Model.StructureSelectionFromBundle,
             {bundle: bundle, label: "binding site"})
      .apply(ST.Representation.StructureRepresentation3D, {
        type: {name: "ball-and-stick",
               params: {sizeFactor: this.siteSize, aromaticBonds: false,
                        sizeAspectRatio: 0.5}},
        colorTheme: {name: "element-symbol",
                     params: {carbonColor: {name: "element-symbol",
                                            params: {}}}},
        sizeTheme: {name: "physical", params: {}}
      });
    await update.commit();
    this.site = update.selector ? update.selector.ref : null;
    if (!this.site) {
      // find it back by label so a later clear still works
      this.site = null;
    }
    return found;
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
                 color: COLORS[l.kind] === undefined ? 0x8b93a3 : COLORS[l.kind],
                 dash: l.quality >= 0.55 ? 0.1 : 0.45,
                 size: (0.08 + 0.1 * l.quality) * self.lineSize,
                 text: l.kind.replace(/_/g, " ") + "  "
                       + l.distance.toFixed(2) + " Å  q "
                       + l.quality.toFixed(2)});
    });
    if (opts.clashes !== false) {
      (scene.clashes || []).forEach(function (c) {
        rows.push({a: c.a, b: c.b, color: c.severe ? SEVERE : CLASH,
                   dash: 0.2, size: 0.12 * self.lineSize,
                   text: "clash " + c.overlap.toFixed(2) + " Å"});
      });
    }

    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      var ha = this.atomAt(r.a), hb = this.atomAt(r.b);
      if (!ha || !hb) { missed++; continue; }
      try {
        var cell = await mm.addDistance(lociFor(ha), lociFor(hb), {
          visualParams: {
            linesColor: r.color, linesSize: r.size, dashLength: r.dash,
            textColor: r.color, textSize: 0.33,
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
    return drawn;
  };

  // ------------------------------------------------------------- click

  Viewer.prototype._wireClick = function () {
    var self = this;
    var SP = global.molstar.lib.structure.StructureProperties;
    var SE = global.molstar.lib.structure.StructureElement;
    this.plugin.behaviors.interaction.click.subscribe(function (e) {
      if (!self.onAtomClick) return;
      var loci = e.current && e.current.loci;
      if (!loci || loci.kind !== "element-loci") return;
      try {
        var l = SE.Loci.getFirstLocation(loci);
        if (!l) return;
        self.onAtomClick([SP.atom.x(l), SP.atom.y(l), SP.atom.z(l)],
                         SP.atom.label_atom_id(l));
      } catch (err) { /* a click on nothing is not an error */ }
    });
  };

  Viewer.prototype.focusPose = function () {
    if (!this.pose) return;
    this.plugin.managers.camera.focusLoci(
      global.molstar.lib.structure.Structure.Loci(this.pose.data));
  };

  global.HPViewer = {Viewer: Viewer, COLORS: COLORS};
})(window);

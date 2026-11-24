"""SDF records in the V3000 dialect are read, and read the same as V2000.

RDKit writes V3000 on request and whenever a molecule needs it. The
reader used to know only V2000, where such a record's legacy counts line
says "0 atoms" - so a whole V3000 file parsed to nothing, and the web
page answered the next click with "load poses first".
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hitpicking.io.sdf import read_sdf  # noqa: E402

V2000 = """methylammonium
  test

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4700    0.0000    0.0000 N   0  3  0  0  0  0  0  0  0  0  0  0
    1.8100    0.9500    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8100   -0.4800    0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8100   -0.4800   -0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  2  4  1  0
  2  5  1  0
M  CHG  1   2   1
M  END
> <score>
-7.5

$$$$
"""

V3000 = """methylammonium
  test

  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 5 4 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C 0.0000 0.0000 0.0000 0
M  V30 2 N 1.4700 0.0000 0.0000 0 -
M  V30 CHG=1
M  V30 3 H 1.8100 0.9500 0.0000 0
M  V30 4 H 1.8100 -0.4800 0.8200 0
M  V30 5 H 1.8100 -0.4800 -0.8200 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 2 3
M  V30 3 1 2 4
M  V30 4 1 2 5
M  V30 END BOND
M  V30 END CTAB
M  END
> <score>
-7.5

$$$$
"""


def _read(tmp_path, name, text, copies=1):
    p = tmp_path / name
    p.write_text(text * copies)
    return read_sdf(str(p))


def test_v3000_is_read_at_all(tmp_path):
    mols = _read(tmp_path, "a.sdf", V3000, copies=3)
    assert len(mols) == 3
    assert mols[0].n_atoms == 5


def test_v3000_matches_v2000(tmp_path):
    a = _read(tmp_path, "a.sdf", V2000)[0]
    b = _read(tmp_path, "b.sdf", V3000)[0]
    assert list(a.element) == list(b.element)
    assert np.allclose(a.xyz, b.xyz)
    assert sorted(map(tuple, a.bonds.tolist())) == sorted(map(tuple, b.bonds.tolist()))
    assert list(a.formal_charge) == list(b.formal_charge) == [0, 1, 0, 0, 0]
    assert a.meta["props"] == b.meta["props"] == {"score": -7.5}


def test_a_file_can_mix_the_two(tmp_path):
    mols = _read(tmp_path, "m.sdf", V2000 + V3000)
    assert [m.n_atoms for m in mols] == [5, 5]

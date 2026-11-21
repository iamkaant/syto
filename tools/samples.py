"""Where the sample data lives.

Most of the suite needs a real receptor and a real set of docked poses -
golden-file tests on invented molecules would pin the wrong thing. That
data is not in this repository: it is large, and it is not ours to
redistribute. So the tests look for a checkout of AutoParty.

The path used to be written into twenty-odd files as one laptop's home
directory, which meant the suite ran on exactly one machine and silently
skipped everywhere else - including, eventually, here.

Resolution order:

  1. $HITPICK_SAMPLES, if set - point it at an AutoParty checkout
  2. a sibling of this repository, which is the usual layout
  3. ./samples inside this repository

A test that cannot find the data skips. It does not fail: not having
someone else's dataset is not a defect in this code.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDIDATES = (
    os.environ.get("HITPICK_SAMPLES") or "",
    os.path.join(os.path.dirname(ROOT), "AutoParty"),
    os.path.join(ROOT, "samples"),
)

RECEPTOR_REL = os.path.join("autoparty", "inputs", "rec.crg.pdb")
POSES_REL = os.path.join("autoparty", "inputs",
                         "more_opcore_enum_asp_h2o_hbond.mol2")


def autoparty():
    """The first candidate that actually holds the sample receptor."""
    for c in CANDIDATES:
        if c and os.path.exists(os.path.join(c, RECEPTOR_REL)):
            return c
    return ""


AUTOPARTY = autoparty()
RECEPTOR = os.path.join(AUTOPARTY, RECEPTOR_REL) if AUTOPARTY else ""
POSES = os.path.join(AUTOPARTY, POSES_REL) if AUTOPARTY else ""


def available():
    return bool(AUTOPARTY)


def why_not():
    return ("sample data not found. Set HITPICK_SAMPLES to an AutoParty "
            "checkout, or put one beside this repository. Looked in: %s"
            % ", ".join(c for c in CANDIDATES if c))

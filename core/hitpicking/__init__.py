"""hitpicking -- shared engine for the hit-picking helper.

One pure-Python + numpy core, used unchanged by:
  * the ChimeraX bundle (Python 3.11, no RDKit available)
  * the static web page (via Pyodide, no Python at all in the browser)
  * the optional local sidecar

Nothing in this package may import RDKit, ChimeraX, torch, pandas or scipy.
Those belong in adapters and optional extras. See docs/ARCHITECTURE.md.
"""

__version__ = "0.1.0.dev0"

# Feature-vector schema version for the active-learning module. Models trained
# under one version must not be loaded under another.
FEATURE_VERSION = 2

# Handbook data schema version.
HANDBOOK_SCHEMA_VERSION = 1

from .molsys import (  # noqa: E402,F401
    MolSystem,
    HYB_UNKNOWN, HYB_SP, HYB_SP2, HYB_SP3,
    PART_RECEPTOR, PART_LIGAND, PART_WATER, PART_ION, PART_COFACTOR,
)

__all__ = [
    "MolSystem",
    "HYB_UNKNOWN", "HYB_SP", "HYB_SP2", "HYB_SP3",
    "PART_RECEPTOR", "PART_LIGAND", "PART_WATER", "PART_ION", "PART_COFACTOR",
    "FEATURE_VERSION", "HANDBOOK_SCHEMA_VERSION", "__version__",
]

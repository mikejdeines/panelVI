"""MERFISH gene panel selection with sparse hierarchical scANVI.

Model and training APIs are in :mod:`panelvi.hierarchical_scanvi_panel`.
Importing this package alone does not initialize the scientific Python stack.
"""

__version__ = "1.0.0"

from .create_annbatch_collection import create_annbatch_collection
from .run_panel_selection import run_panel_selection

__all__ = ["create_annbatch_collection", "run_panel_selection"]

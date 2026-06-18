"""wcpredictor: a self-learning Monte Carlo World Cup predictor engine.

Public API re-exports for convenience.
"""

from .config import Params, Paths
from .ratings import Rating, RatingStore
from .simulate import run_simulation

__version__ = "0.1.0"

__all__ = [
    "Params",
    "Paths",
    "Rating",
    "RatingStore",
    "run_simulation",
    "__version__",
]

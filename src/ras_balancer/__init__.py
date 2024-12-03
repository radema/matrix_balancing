"""
RAS Balancer - A Python library for matrix balancing using the RAS and GRAS algorithms.

This package provides tools for:
- Matrix balancing using the RAS (Iterative Proportional Fitting) algorithm
- Matrix balancing using the GRAS algorithm for mixed-sign matrices
- Generation of balanced matrices for testing and simulation
- Application of various types of shocks to balanced matrices
"""

__version__ = "0.1.0"
__author__ = "Raul De Maio"
__license__ = "MIT"
__all__ = ["core", "generator", "shocker", "types"]

from . import core
from . import generator
from . import shocker
from . import types

# Explicitly import relevant classes and functions for top-level access
from .core import RASBalancer, GRASBalancer, balance_matrix  # noqa: F401
from .generator import MatrixGenerator  # noqa: F401
from .shocker import MatrixShocker, ShockType  # noqa: F401
from .types import BalanceStatus, RASResult, ShockResult, BalanceCheckResult, MatrixGenerationResult  # noqa: F401

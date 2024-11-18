"""
RAS Balancer - A Python library for matrix balancing using the RAS algorithm.

This package provides tools for:
- Matrix balancing using the RAS (Iterative Proportional Fitting) algorithm
- Generation of balanced matrices for testing and simulation
- Application of various types of shocks to balanced matrices
"""

__version__ = "0.0.1"
__author__ = "Raul De Maio"
__license__ = "MIT"
__all__ = ["core", "generator", "shocker", "types"]

from .core import *
from .generator import *
from .shocker import *
from .types import *
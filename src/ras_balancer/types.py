# types.py
"""Type definitions and dataclasses for the RAS balancing library."""

from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from dataclasses import dataclass
from enum import Enum


class BalanceStatus(Enum):
    """Enumeration for matrix balance status."""

    BALANCED = "balanced"
    UNBALANCED_ROWS = "unbalanced_rows"
    UNBALANCED_COLS = "unbalanced_cols"
    UNBALANCED_BOTH = "unbalanced_both"
    INVALID = "invalid"


class ShockType(Enum):
    """Enumeration for different types of matrix shocks."""

    CELL = "cell"
    ROW_TOTAL = "row_total"
    COLUMN_TOTAL = "column_total"
    PROPORTIONAL = "proportional"
    RANDOM = "random"


@dataclass
class BalanceCheckResult:
    """Data class to store balance check results."""

    status: BalanceStatus
    max_row_deviation: float
    max_col_deviation: float
    row_deviations: NDArray
    col_deviations: NDArray
    is_balanced: bool

    def __str__(self) -> str:
        return (
            f"Balance Check Result:\n"
            f"Status: {self.status.value}\n"
            f"Maximum Row Deviation: {self.max_row_deviation:.2e}\n"
            f"Maximum Column Deviation: {self.max_col_deviation:.2e}\n"
            f"Is Balanced: {self.is_balanced}"
        )


@dataclass
class RASResult:
    """Data class to store RAS algorithm results."""

    balanced_matrix: Union[NDArray, sp.spmatrix]
    iterations: int
    converged: bool
    row_error: float
    col_error: float

    def __str__(self) -> str:
        return (
            f"RAS Result:\n"
            f"Converged: {self.converged} in {self.iterations} iterations\n"
            f"Final errors - Row: {self.row_error:.2e}, Column: {self.col_error:.2e}"
        )


@dataclass
class ShockResult:
    """Data class to store results of matrix shock application."""

    original_matrix: Union[NDArray, sp.spmatrix]
    shocked_matrix: Union[NDArray, sp.spmatrix]
    original_row_sums: NDArray
    original_col_sums: NDArray
    new_row_sums: NDArray
    new_col_sums: NDArray
    shock_type: ShockType
    shock_magnitude: float
    affected_indices: Union[Tuple[int, int], NDArray]

    def __str__(self) -> str:
        return (
            f"Shock Result:\n"
            f"Type: {self.shock_type.value}\n"
            f"Magnitude: {self.shock_magnitude:.2e}\n"
            f"Max Row Sum Change: {np.max(np.abs(self.new_row_sums - self.original_row_sums)):.2e}"
            f"Max Col Sum Change: {np.max(np.abs(self.new_col_sums - self.original_col_sums)):.2e}"
        )

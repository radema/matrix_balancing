# shocker.py
"""Functionality for introducing controlled shocks to balanced matrices."""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .core import balance_matrix
from .types import ShockResult, ShockType


class MatrixShocker:
    """Class for introducing controlled shocks to balanced matrices."""

    def __init__(self, preserve_zeros: bool = True, random_seed: Optional[int] = None):
        """
        Initialize the MatrixShocker.

        Parameters
        ----------
        preserve_zeros : bool
            If True, shock operations won't modify zero elements.
        random_seed : Optional[int]
            Seed for random number generation.
        """
        self.preserve_zeros = preserve_zeros
        if random_seed is not None:
            np.random.seed(random_seed)

    def shock_cell(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        row: int,
        col: int,
        magnitude: float,
        relative: bool = True,
    ) -> ShockResult:
        """
        Introduce a shock to a specific cell in the matrix.
        """
        shocked = matrix.copy()
        original_value = shocked[row, col]

        if self.preserve_zeros and original_value == 0:
            warnings.warn(f"Cannot shock zero value at ({row}, {col}) when preserve_zeros is True.")
            return self._create_shock_result(matrix, shocked, ShockType.CELL, magnitude, (row, col))

        shocked[row, col] = (
            original_value * (1 + magnitude) if relative else original_value + magnitude
        )

        return self._create_shock_result(matrix, shocked, ShockType.CELL, magnitude, (row, col))

    def shock_row_totals(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        row_indices: List[int],
        magnitudes: List[float],
        relative: bool = True,
    ) -> ShockResult:
        """
        Shock row totals while preserving relative proportions within rows.
        """
        shocked = matrix.copy()
        for idx, magnitude in zip(row_indices, magnitudes):
            row_values = shocked[idx].toarray().ravel() if sp.issparse(shocked) else shocked[idx]
            current_sum = row_values.sum()

            if relative:
                target_sum = current_sum * (1 + magnitude)
            else:
                target_sum = current_sum + magnitude

            scaling_factor = target_sum / current_sum if current_sum != 0 else 1
            if self.preserve_zeros:
                nonzero_mask = row_values != 0
                row_values[nonzero_mask] *= scaling_factor
            else:
                row_values *= scaling_factor

            if sp.issparse(shocked):
                shocked[idx] = sp.csr_matrix(row_values)
            else:
                shocked[idx] = row_values

        return self._create_shock_result(
            matrix, shocked, ShockType.ROW_TOTAL, np.mean(magnitudes), row_indices
        )

    def shock_column_totals(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        col_indices: List[int],
        magnitudes: List[float],
        relative: bool = True,
    ) -> ShockResult:
        """
        Shock column totals while preserving relative proportions within columns.
        """
        shocked = matrix.transpose() if sp.issparse(matrix) else matrix.T
        result = self.shock_row_totals(shocked, col_indices, magnitudes, relative)
        shocked = (
            result.shocked_matrix.transpose()
            if sp.issparse(result.shocked_matrix)
            else result.shocked_matrix.T
        )

        return self._create_shock_result(
            matrix, shocked, ShockType.COLUMN_TOTAL, np.mean(magnitudes), col_indices
        )

    def _select_shock_indices(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        n_shocks: int,
    ) -> Tuple[NDArray, NDArray]:
        """Select indices to shock based on matrix type and preservation settings."""
        should_preserve = sp.issparse(matrix) or self.preserve_zeros

        if should_preserve:
            if sp.issparse(matrix):
                rows, cols = matrix.nonzero()
            else:
                rows, cols = np.where(matrix != 0)

            n_candidates = len(rows)
            if n_candidates == 0:
                return np.array([], dtype=int), np.array([], dtype=int)

            count = min(n_shocks, n_candidates)
            indices = np.random.choice(n_candidates, count, replace=False)
            return rows[indices], cols[indices]
        else:
            rows, cols = matrix.shape
            n_total = rows * cols
            indices = np.random.choice(n_total, min(n_shocks, n_total), replace=False)
            return indices // cols, indices % cols

    def shock_proportional(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        magnitude: float,
        affected_fraction: float = 0.1,
    ) -> ShockResult:
        """
        Introduce proportional shocks to random elements while preserving structure.
        """
        shocked = matrix.copy()

        # Calculate number of shocks
        if sp.issparse(matrix):
            n_elements = matrix.nnz
        elif self.preserve_zeros:
            n_elements = np.count_nonzero(matrix)
        else:
            n_elements = matrix.size

        n_shocks = int(n_elements * affected_fraction)

        shock_rows, shock_cols = self._select_shock_indices(matrix, n_shocks)

        shock_magnitudes = np.random.uniform(-magnitude, magnitude, len(shock_rows))
        for row, col, value in zip(shock_rows, shock_cols, shock_magnitudes):
            shocked[row, col] *= 1 + value

        return self._create_shock_result(
            matrix,
            shocked,
            ShockType.PROPORTIONAL,
            magnitude,
            np.column_stack((shock_rows, shock_cols)),
        )

    def shock_random(
        self, matrix: Union[NDArray, sp.spmatrix], magnitude: float, n_shocks: int
    ) -> ShockResult:
        """
        Introduce random shocks to specific cells.
        """
        shocked = matrix.copy()

        shock_rows, shock_cols = self._select_shock_indices(matrix, n_shocks)

        shock_values = np.random.uniform(-magnitude, magnitude, len(shock_rows))
        for row, col, value in zip(shock_rows, shock_cols, shock_values):
            shocked[row, col] += value

        return self._create_shock_result(
            matrix, shocked, ShockType.RANDOM, magnitude, np.column_stack((shock_rows, shock_cols))
        )

    def shock_and_rebalance(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        row_sums: NDArray,
        col_sums: NDArray,
        method: str = "RAS",
        **kwargs,
    ) -> ShockResult:
        """
        Apply shocks and rebalance the matrix using the specified method.
        """
        shocked = self.shock_proportional(
            matrix, magnitude=0.1, affected_fraction=0.05
        ).shocked_matrix
        rebalanced = balance_matrix(
            shocked, row_sums, col_sums, method=method, **kwargs
        ).balanced_matrix

        return self._create_shock_result(matrix, rebalanced, ShockType.PROPORTIONAL, 0.1, None)

    @staticmethod
    def _calculate_sums(matrix: Union[NDArray, sp.spmatrix]) -> Tuple[NDArray, NDArray]:
        """Calculate row and column sums for sparse or dense matrix."""
        if sp.issparse(matrix):
            row_sums = matrix.sum(axis=1).A1
            col_sums = matrix.sum(axis=0).A1
        else:
            row_sums = matrix.sum(axis=1)
            col_sums = matrix.sum(axis=0)
        return row_sums, col_sums

    def _create_shock_result(
        self,
        original: Union[NDArray, sp.spmatrix],
        shocked: Union[NDArray, sp.spmatrix],
        shock_type: ShockType,
        magnitude: float,
        affected_indices: Union[Tuple[int, int], NDArray],
    ) -> ShockResult:
        """Create a ShockResult object from shocked matrix data."""
        orig_row_sums, orig_col_sums = self._calculate_sums(original)
        new_row_sums, new_col_sums = self._calculate_sums(shocked)

        return ShockResult(
            original_matrix=original,
            shocked_matrix=shocked,
            original_row_sums=orig_row_sums,
            original_col_sums=orig_col_sums,
            new_row_sums=new_row_sums,
            new_col_sums=new_col_sums,
            shock_type=shock_type,
            shock_magnitude=magnitude,
            affected_indices=affected_indices,
        )

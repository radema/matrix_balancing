# shocker.py
"""Functionality for introducing controlled shocks to balanced matrices."""

import numpy as np
import scipy.sparse as sp
import warnings
from typing import Union, Optional, List, Tuple
from numpy.typing import NDArray
from .types import ShockType, ShockResult


class MatrixShocker:
    """Class for introducing controlled shocks to balanced matrices."""

    def __init__(self, preserve_zeros: bool = True, random_seed: Optional[int] = None):
        """
        Initialize the MatrixShocker.

        Parameters
        ----------
        preserve_zeros : bool
            If True, shock operations won't modify zero elements
        random_seed : Optional[int]
            Seed for random number generation
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

        Parameters
        ----------
        matrix : Union[NDArray, sp.spmatrix]
            Input matrix to shock
        row : int
            Row index of cell to shock
        col : int
            Column index of cell to shock
        magnitude : float
            Shock magnitude (as multiplier if relative=True, absolute value if relative=False)
        relative : bool
            If True, magnitude is treated as a multiplier

        Returns
        -------
        ShockResult
            Results of shock application
        """
        shocked = matrix.copy()
        original_value = shocked[row, col]

        if self.preserve_zeros and original_value == 0:
            warnings.warn(f"Cannot shock zero value at ({row}, {col}) when preserve_zeros is True")
            return self._create_shock_result(matrix, shocked, ShockType.CELL, magnitude, (row, col))

        if relative:
            shocked[row, col] = original_value * (1 + magnitude)
        else:
            shocked[row, col] = original_value + magnitude

        return self._create_shock_result(matrix, shocked, ShockType.CELL, magnitude, (row, col))

    def shock_row_totals(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        row_indices: Union[int, List[int], NDArray],
        magnitudes: Union[float, List[float], NDArray],
        relative: bool = True,
        distribution: str = "proportional",
    ) -> ShockResult:
        """
        Shock row totals while preserving relative proportions within rows.

        Parameters
        ----------
        matrix : Union[NDArray, sp.spmatrix]
            Input matrix to shock
        row_indices : Union[int, List[int], NDArray]
            Indices of rows to shock
        magnitudes : Union[float, List[float], NDArray]
            Shock magnitudes for each row
        relative : bool
            If True, magnitudes are treated as multipliers
        distribution : str
            How to distribute the shock ('proportional' or 'uniform')

        Returns
        -------
        ShockResult
            Results of shock application
        """
        shocked = matrix.copy()
        row_indices = np.atleast_1d(row_indices)
        magnitudes = np.atleast_1d(magnitudes)

        if len(magnitudes) == 1:
            magnitudes = np.repeat(magnitudes, len(row_indices))

        for row_idx, magnitude in zip(row_indices, magnitudes):
            row_values = (
                shocked[row_idx].toarray().ravel() if sp.issparse(shocked) else shocked[row_idx]
            )
            current_sum = row_values.sum()

            if relative:
                target_sum = current_sum * (1 + magnitude)
            else:
                target_sum = current_sum + magnitude

            if distribution == "proportional":
                scaling_factor = target_sum / current_sum
                if self.preserve_zeros:
                    nonzero_mask = row_values != 0
                    row_values[nonzero_mask] *= scaling_factor
                else:
                    row_values *= scaling_factor
            else:  # uniform
                if self.preserve_zeros:
                    nonzero_mask = row_values != 0
                    addition = (target_sum - current_sum) / np.sum(nonzero_mask)
                    row_values[nonzero_mask] += addition
                else:
                    addition = (target_sum - current_sum) / len(row_values)
                    row_values += addition

            if sp.issparse(shocked):
                shocked[row_idx] = sp.csr_matrix(row_values)
            else:
                shocked[row_idx] = row_values

        return self._create_shock_result(
            matrix, shocked, ShockType.ROW_TOTAL, np.mean(magnitudes), row_indices
        )

    def shock_column_totals(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        col_indices: Union[int, List[int], NDArray],
        magnitudes: Union[float, List[float], NDArray],
        relative: bool = True,
        distribution: str = "proportional",
    ) -> ShockResult:
        """
        Shock column totals while preserving relative proportions within columns.

        Parameters
        ----------
        matrix : Union[NDArray, sp.spmatrix]
            Input matrix to shock
        col_indices : Union[int, List[int], NDArray]
            Indices of columns to shock
        magnitudes : Union[float, List[float], NDArray]
            Shock magnitudes for each column
        relative : bool
            If True, magnitudes are treated as multipliers
        distribution : str
            How to distribute the shock ('proportional' or 'uniform')

        Returns
        -------
        ShockResult
            Results of shock application
        """
        # Transpose, shock rows, transpose back
        shocked = matrix.transpose() if sp.issparse(matrix) else matrix.T
        result = self.shock_row_totals(shocked, col_indices, magnitudes, relative, distribution)
        shocked = (
            result.shocked_matrix.transpose()
            if sp.issparse(result.shocked_matrix)
            else result.shocked_matrix.T
        )

        return self._create_shock_result(
            matrix, shocked, ShockType.COLUMN_TOTAL, np.mean(magnitudes), col_indices
        )

    def shock_proportional(
        self, matrix: Union[NDArray, sp.spmatrix], magnitude: float, affected_fraction: float = 0.1
    ) -> ShockResult:
        """
        Introduce proportional shocks to random elements while preserving structure.

        Parameters
        ----------
        matrix : Union[NDArray, sp.spmatrix]
            Input matrix to shock
        magnitude : float
            Maximum shock magnitude (as proportion)
        affected_fraction : float
            Fraction of non-zero elements to shock

        Returns
        -------
        ShockResult
            Results of shock application
        """
        shocked = matrix.copy()

        if sp.issparse(matrix):
            nnz_rows, nnz_cols = matrix.nonzero()
            n_shocks = int(len(nnz_rows) * affected_fraction)
            shock_indices = np.random.choice(len(nnz_rows), n_shocks, replace=False)
            shock_rows = nnz_rows[shock_indices]
            shock_cols = nnz_cols[shock_indices]

            for row, col in zip(shock_rows, shock_cols):
                shock_magnitude = np.random.uniform(-magnitude, magnitude)
                shocked[row, col] *= 1 + shock_magnitude
        else:
            if self.preserve_zeros:
                nonzero_mask = matrix != 0
                n_nonzero = np.sum(nonzero_mask)
                n_shocks = int(n_nonzero * affected_fraction)
                shock_indices = np.random.choice(n_nonzero, n_shocks, replace=False)
                nonzero_positions = np.where(nonzero_mask)
                shock_rows = nonzero_positions[0][shock_indices]
                shock_cols = nonzero_positions[1][shock_indices]
            else:
                n_elements = matrix.size
                n_shocks = int(n_elements * affected_fraction)
                shock_indices = np.random.choice(n_elements, n_shocks, replace=False)
                shock_rows = shock_indices // matrix.shape[1]
                shock_cols = shock_indices % matrix.shape[1]

            shock_magnitudes = np.random.uniform(-magnitude, magnitude, n_shocks)
            shocked[shock_rows, shock_cols] *= 1 + shock_magnitudes

        return self._create_shock_result(
            matrix,
            shocked,
            ShockType.PROPORTIONAL,
            magnitude,
            np.column_stack((shock_rows, shock_cols)),
        )

    def shock_random(
        self, matrix: Union[NDArray, sp.spmatrix], magnitude: float, n_shocks: int = 1
    ) -> ShockResult:
        """
        Introduce random shocks to specific cells.

        Parameters
        ----------
        matrix : Union[NDArray, sp.spmatrix]
            Input matrix to shock
        magnitude : float
            Maximum absolute shock value
        n_shocks : int
            Number of cells to shock

        Returns
        -------
        ShockResult
            Results of shock application
        """
        shocked = matrix.copy()
        rows, cols = matrix.shape

        if sp.issparse(matrix):
            nnz_rows, nnz_cols = matrix.nonzero()
            if len(nnz_rows) < n_shocks:
                n_shocks = len(nnz_rows)
            shock_indices = np.random.choice(len(nnz_rows), n_shocks, replace=False)
            shock_rows = nnz_rows[shock_indices]
            shock_cols = nnz_cols[shock_indices]
        else:
            if self.preserve_zeros:
                nonzero_mask = matrix != 0
                nonzero_positions = np.where(nonzero_mask)
                if len(nonzero_positions[0]) < n_shocks:
                    n_shocks = len(nonzero_positions[0])
                shock_indices = np.random.choice(len(nonzero_positions[0]), n_shocks, replace=False)
                shock_rows = nonzero_positions[0][shock_indices]
                shock_cols = nonzero_positions[1][shock_indices]
            else:
                shock_rows = np.random.randint(0, rows, n_shocks)
                shock_cols = np.random.randint(0, cols, n_shocks)

        shock_values = np.random.uniform(-magnitude, magnitude, n_shocks)

        for row, col, value in zip(shock_rows, shock_cols, shock_values):
            shocked[row, col] += value

        return self._create_shock_result(
            matrix, shocked, ShockType.RANDOM, magnitude, np.column_stack((shock_rows, shock_cols))
        )

    @staticmethod
    def _create_shock_result(
        original: Union[NDArray, sp.spmatrix],
        shocked: Union[NDArray, sp.spmatrix],
        shock_type: ShockType,
        magnitude: float,
        affected_indices: Union[Tuple[int, int], NDArray],
    ) -> ShockResult:
        """Create a ShockResult object from shocked matrix data."""
        if sp.issparse(original):
            orig_row_sums = original.sum(axis=1).A1
            orig_col_sums = original.sum(axis=0).A1
            new_row_sums = shocked.sum(axis=1).A1
            new_col_sums = shocked.sum(axis=0).A1
        else:
            orig_row_sums = original.sum(axis=1)
            orig_col_sums = original.sum(axis=0)
            new_row_sums = shocked.sum(axis=1)
            new_col_sums = shocked.sum(axis=0)

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

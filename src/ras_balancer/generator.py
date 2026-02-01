# generator.py
"""Utilities for generating random balanced matrices."""

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .core import balance_matrix


class MatrixGenerator:
    """Class for generating random balanced matrices."""

    @staticmethod
    def _generate_random_positive_vector(size: int, total: float = 1.0) -> NDArray:
        """Generate a random positive vector that sums to a specified total."""
        vector = np.random.uniform(0.1, 1.0, size)
        return vector * (total / vector.sum())

    @classmethod
    def generate_balanced_dense(
        cls,
        rows: int,
        cols: int,
        total_sum: float = 1000.0,
        method: str = "RAS",
        noise_level: float = 0.1,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Generate a random balanced dense matrix using RAS or GRAS.

        Parameters
        ----------
        rows : int
            Number of rows in the matrix.
        cols : int
            Number of columns in the matrix.
        total_sum : float, optional
            Total sum of the matrix, by default 1000.0.
        method : str, optional
            Balancing method ('RAS' or 'GRAS'), by default 'RAS'.
        noise_level : float, optional
            Level of noise to add before balancing, by default 0.1.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            Balanced matrix, row sums, and column sums.
        """
        row_sums = cls._generate_random_positive_vector(rows, total_sum)
        col_sums = cls._generate_random_positive_vector(cols, total_sum)

        row_structure = row_sums.reshape(-1, 1)
        col_structure = col_sums.reshape(1, -1)
        base_matrix = np.sqrt(np.outer(row_structure, col_structure))

        if noise_level > 0:
            noise = np.random.uniform(1 - noise_level, 1 + noise_level, (rows, cols))
            base_matrix *= noise

        # Balance the matrix using the specified method
        balanced_matrix = balance_matrix(
            base_matrix,
            row_sums,
            col_sums,
            method=method,
            tolerance=1e-6,
            max_iter=2000,
        ).balanced_matrix

        return balanced_matrix, row_sums, col_sums

    @classmethod
    def generate_balanced_sparse(
        cls,
        rows: int,
        cols: int,
        density: float = 0.01,
        total_sum: float = 1000.0,
        method: str = "RAS",
    ) -> Tuple[sp.spmatrix, NDArray, NDArray]:
        """
        Generate a random balanced sparse matrix using RAS or GRAS.

        Parameters
        ----------
        rows : int
            Number of rows in the matrix.
        cols : int
            Number of columns in the matrix.
        density : float, optional
            Fraction of non-zero elements, by default 0.01.
        total_sum : float, optional
            Total sum of the matrix, by default 1000.0.
        method : str, optional
            Balancing method ('RAS' or 'GRAS'), by default 'RAS'.

        Returns
        -------
        Tuple[sp.spmatrix, NDArray, NDArray]
            Balanced sparse matrix, row sums, and column sums.
        """
        if not (0 <= density <= 1):
            raise ValueError("Density must be between 0 and 1 inclusive.")

        row_sums = cls._generate_random_positive_vector(rows, total_sum)
        col_sums = cls._generate_random_positive_vector(cols, total_sum)

        nnz = int(rows * cols * density)
        row_indices = np.random.choice(rows, nnz)
        col_indices = np.random.choice(cols, nnz)
        data = np.random.uniform(0.1, 1.0, nnz)

        matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(rows, cols))

        # Balance the matrix using the specified method
        balanced_matrix = balance_matrix(
            matrix,
            row_sums,
            col_sums,
            method=method,
            tolerance=1e-6,
            max_iter=2000,
        ).balanced_matrix

        return balanced_matrix, row_sums, col_sums

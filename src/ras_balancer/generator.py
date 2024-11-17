# generator.py
"""Utilities for generating random balanced matrices."""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Union
from numpy.typing import NDArray
from .core import RASBalancer

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
        min_value: float = 0.1,
        noise_level: float = 0.1
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Generate a random balanced dense matrix."""
        row_sums = cls._generate_random_positive_vector(rows, total_sum)
        col_sums = cls._generate_random_positive_vector(cols, total_sum)
        
        row_structure = row_sums.reshape(-1, 1)
        col_structure = col_sums.reshape(1, -1)
        base_matrix = np.sqrt(np.outer(row_structure, col_structure))
        
        if noise_level > 0:
            noise = np.random.uniform(1 - noise_level, 1 + noise_level, (rows, cols))
            base_matrix *= noise
            
            balancer = RASBalancer(tolerance=1e-10)
            result = balancer.balance(base_matrix, row_sums, col_sums)
            base_matrix = result.balanced_matrix
        
        return base_matrix, row_sums, col_sums
    
    @classmethod
    def generate_balanced_sparse(
        cls,
        rows: int,
        cols: int,
        density: float = 0.01,
        total_sum: float = 1000.0,
        min_value: float = 0.1
    ) -> Tuple[sp.spmatrix, NDArray, NDArray]:
        """Generate a random balanced sparse matrix."""
        row_sums = cls._generate_random_positive_vector(rows, total_sum)
        col_sums = cls._generate_random_positive_vector(cols, total_sum)
        
        nnz = int(rows * cols * density)
        row_indices = np.random.choice(rows, nnz)
        col_indices = np.random.choice(cols, nnz)
        data = np.random.uniform(min_value, 1.0, nnz)
        
        matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(rows, cols))
        
        balancer = RASBalancer(tolerance=1e-10, use_sparse=True)
        result = balancer.balance(matrix, row_sums, col_sums)
        
        return result.balanced_matrix, row_sums, col_sums

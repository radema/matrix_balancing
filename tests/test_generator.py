import unittest
import numpy as np
import scipy.sparse as sp
from ras_library import (
    MatrixGenerator
)

class TestMatrixGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.rows = 100
        self.cols = 100
        self.total_sum = 1000.0

    def test_generate_balanced_dense(self):
        """Test generation of balanced dense matrices."""
        matrix, row_sums, col_sums = MatrixGenerator.generate_balanced_dense(
            self.rows,
            self.cols,
            total_sum=self.total_sum
        )
        
        # Check dimensions
        self.assertEqual(matrix.shape, (self.rows, self.cols))
        
        # Check if matrix is balanced
        actual_row_sums = matrix.sum(axis=1)
        actual_col_sums = matrix.sum(axis=0)
        
        self.assertTrue(np.allclose(actual_row_sums, row_sums, rtol=1e-6))
        self.assertTrue(np.allclose(actual_col_sums, col_sums, rtol=1e-6))
        self.assertTrue(np.allclose(matrix.sum(), self.total_sum, rtol=1e-6))

    def test_generate_balanced_sparse(self):
        """Test generation of balanced sparse matrices."""
        density = 0.01
        matrix, row_sums, col_sums = MatrixGenerator.generate_balanced_sparse(
            self.rows,
            self.cols,
            density=density,
            total_sum=self.total_sum
        )
        
        # Check if result is sparse
        self.assertTrue(sp.issparse(matrix))
        
        # Check sparsity
        actual_density = matrix.nnz / (self.rows * self.cols)
        self.assertLess(abs(actual_density - density), 0.01)
        
        # Check balance
        actual_row_sums = matrix.sum(axis=1).A1
        actual_col_sums = matrix.sum(axis=0).A1
        
        self.assertTrue(np.allclose(actual_row_sums, row_sums, rtol=1e-6))
        self.assertTrue(np.allclose(actual_col_sums, col_sums, rtol=1e-6))
        self.assertTrue(np.allclose(matrix.sum(), self.total_sum, rtol=1e-6))

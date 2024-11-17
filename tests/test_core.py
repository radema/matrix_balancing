import unittest
import numpy as np
import scipy.sparse as sp
from ras_library import (
    RASBalancer
)


class TestRASBalancer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.rows = 5
        self.cols = 4
        self.balancer = RASBalancer(max_iter=100, tolerance=1e-8)
        
        # Create a simple test matrix
        self.test_matrix = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0]
        ])
        self.target_row_sums = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        self.target_col_sums = np.array([12.5, 12.5, 12.5, 12.5])

    def test_balance_dense_matrix(self):
        """Test balancing a dense matrix."""
        result = self.balancer.balance(
            self.test_matrix,
            self.target_row_sums,
            self.target_col_sums
        )
        
        # Check convergence
        self.assertTrue(result.converged)
        
        # Check row sums
        row_sums = result.balanced_matrix.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, self.target_row_sums, rtol=1e-6))
        
        # Check column sums
        col_sums = result.balanced_matrix.sum(axis=0)
        self.assertTrue(np.allclose(col_sums, self.target_col_sums, rtol=1e-6))

    def test_balance_sparse_matrix(self):
        """Test balancing a sparse matrix."""
        sparse_matrix = sp.csr_matrix(self.test_matrix)
        result = self.balancer.balance(
            sparse_matrix,
            self.target_row_sums,
            self.target_col_sums
        )
        
        self.assertTrue(result.converged)
        self.assertTrue(sp.issparse(result.balanced_matrix))
        
        row_sums = result.balanced_matrix.sum(axis=1).A1
        col_sums = result.balanced_matrix.sum(axis=0).A1
        
        self.assertTrue(np.allclose(row_sums, self.target_row_sums, rtol=1e-6))
        self.assertTrue(np.allclose(col_sums, self.target_col_sums, rtol=1e-6))

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test mismatched dimensions
        wrong_row_sums = np.ones(self.rows + 1)
        with self.assertRaises(ValueError):
            self.balancer.balance(self.test_matrix, wrong_row_sums, self.target_col_sums)
        
        # Test negative target sums
        negative_sums = -1 * self.target_row_sums
        with self.assertRaises(ValueError):
            self.balancer.balance(self.test_matrix, negative_sums, self.target_col_sums)
        
        # Test unequal total sums
        unequal_col_sums = self.target_col_sums * 2
        with self.assertRaises(ValueError):
            self.balancer.balance(self.test_matrix, self.target_row_sums, unequal_col_sums)

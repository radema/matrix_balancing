import numpy as np
import scipy.sparse as sp
import pytest
from ras_balancer import RASBalancer
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


class TestRASBalancer:
    @pytest.fixture(autouse=True)
    def setup_balancer(self):
        """Set up test fixtures."""
        self.rows = 5
        self.cols = 4
        self.balancer = RASBalancer(max_iter=100, tolerance=1e-8)

        # Create a simple test matrix
        self.test_matrix = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
        self.target_row_sums = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        self.target_col_sums = np.array([12.5, 12.5, 12.5, 12.5])

    def test_balance_dense_matrix(self):
        """Test balancing a dense matrix."""
        test_matrix = self.test_matrix
        target_rows = self.target_row_sums
        target_cols = self.target_col_sums

        result = self.balancer.balance(test_matrix, target_rows, target_cols)

        # Check convergence
        assert result.converged

        # Check row sums
        row_sums = result.balanced_matrix.sum(axis=1)
        assert np.allclose(row_sums, self.target_row_sums, rtol=1e-6)

        # Check column sums
        col_sums = result.balanced_matrix.sum(axis=0)
        assert np.allclose(col_sums, self.target_col_sums, rtol=1e-6)

    def test_balance_sparse_matrix(self):
        """Test balancing a sparse matrix."""
        target_rows = self.target_row_sums
        target_cols = self.target_col_sums

        sparse_matrix = sp.csr_matrix(self.test_matrix)
        result = self.balancer.balance(sparse_matrix, target_rows, target_cols)

        # Check convergence and matrix type
        assert result.converged
        assert sp.issparse(result.balanced_matrix)

        # Check row sums
        row_sums = result.balanced_matrix.sum(axis=1).A1
        assert np.allclose(row_sums, self.target_row_sums, rtol=1e-6)

        # Check column sums
        col_sums = result.balanced_matrix.sum(axis=0).A1
        assert np.allclose(col_sums, self.target_col_sums, rtol=1e-6)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test mismatched dimensions
        wrong_row_sums = np.ones(self.rows + 1)
        with pytest.raises(ValueError):
            test_matrix = self.test_matrix
            target_cols = self.target_col_sums
            self.balancer.balance(test_matrix, wrong_row_sums, target_cols)

        # Test negative target sums
        negative_sums = -1 * self.target_row_sums
        with pytest.raises(ValueError):
            test_matrix = self.test_matrix
            target_cols = self.target_col_sums
            self.balancer.balance(test_matrix, negative_sums, target_cols)

        # Test unequal total sums
        unequal_col_sums = self.target_col_sums * 2
        with pytest.raises(ValueError):
            test_matrix = self.test_matrix
            target_rows = self.target_row_sums
            self.balancer.balance(test_matrix, target_rows, unequal_col_sums)

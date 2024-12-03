import numpy as np
import scipy.sparse as sp
import pytest
from ras_balancer import RASBalancer, GRASBalancer, balance_matrix, RASResult, BalanceStatus


class TestMatrixBalancers:
    @pytest.fixture(autouse=True)
    def setup_balancers(self):
        """Set up test fixtures."""
        self.rows = 5
        self.cols = 4
        self.ras_balancer = RASBalancer(max_iter=100, tolerance=1e-8)
        self.gras_balancer = GRASBalancer(max_iter=100, tolerance=1e-8)

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

    def test_ras_balance_dense_matrix(self):
        """Test balancing a dense matrix with RAS."""
        result = self.ras_balancer.balance(self.test_matrix, self.target_row_sums, self.target_col_sums)

        # Check convergence
        assert result.converged

        # Check row sums
        row_sums = result.balanced_matrix.sum(axis=1)
        assert np.allclose(row_sums, self.target_row_sums, rtol=1e-6)

        # Check column sums
        col_sums = result.balanced_matrix.sum(axis=0)
        assert np.allclose(col_sums, self.target_col_sums, rtol=1e-6)

    def test_ras_balance_sparse_matrix(self):
        """Test balancing a sparse matrix with RAS."""
        sparse_matrix = sp.csr_matrix(self.test_matrix)
        result = self.ras_balancer.balance(sparse_matrix, self.target_row_sums, self.target_col_sums)

        # Check convergence and matrix type
        assert result.converged
        assert sp.issparse(result.balanced_matrix)

        # Check row sums
        row_sums = result.balanced_matrix.sum(axis=1).A1
        assert np.allclose(row_sums, self.target_row_sums, rtol=1e-6)

        # Check column sums
        col_sums = result.balanced_matrix.sum(axis=0).A1
        assert np.allclose(col_sums, self.target_col_sums, rtol=1e-6)

    def test_gras_balance_matrix(self):
        """Test balancing a dense matrix with GRAS."""
        result = self.gras_balancer.balance(self.test_matrix, self.target_row_sums, self.target_col_sums)

        # Check convergence
        assert result.converged

        # Check row sums
        row_sums = result.balanced_matrix.sum(axis=1)
        assert np.allclose(row_sums, self.target_row_sums, rtol=1e-6)

        # Check column sums
        col_sums = result.balanced_matrix.sum(axis=0)
        assert np.allclose(col_sums, self.target_col_sums, rtol=1e-6)

    def test_balance_matrix_function(self):
        """Test the generic balance_matrix function."""
        result = balance_matrix(self.test_matrix, self.target_row_sums, self.target_col_sums, method="RAS")

        assert isinstance(result, RASResult)
        assert result.converged
        assert np.allclose(result.balanced_matrix.sum(axis=1), self.target_row_sums, rtol=1e-6)
        assert np.allclose(result.balanced_matrix.sum(axis=0), self.target_col_sums, rtol=1e-6)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Mismatched dimensions
        wrong_row_sums = np.ones(self.rows + 1)
        with pytest.raises(ValueError):
            self.ras_balancer.balance(self.test_matrix, wrong_row_sums, self.target_col_sums)

        # Negative target sums
        negative_sums = -1 * self.target_row_sums
        with pytest.raises(ValueError):
            self.ras_balancer.balance(self.test_matrix, negative_sums, self.target_col_sums)

        # Unequal total sums
        unequal_col_sums = self.target_col_sums * 2
        with pytest.raises(ValueError):
            self.ras_balancer.balance(self.test_matrix, self.target_row_sums, unequal_col_sums)

    def test_sparse_vs_dense_consistency(self):
        """Test consistency between sparse and dense balancing results."""
        sparse_matrix = sp.csr_matrix(self.test_matrix)
        dense_result = self.ras_balancer.balance(self.test_matrix, self.target_row_sums, self.target_col_sums)
        sparse_result = self.ras_balancer.balance(sparse_matrix, self.target_row_sums, self.target_col_sums)

        assert np.allclose(dense_result.balanced_matrix, sparse_result.balanced_matrix.toarray(), rtol=1e-6)

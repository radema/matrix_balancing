import numpy as np
import pytest
import scipy.sparse as sp

from ras_balancer import MatrixGenerator


class TestMatrixGenerator:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.rows = 100
        self.cols = 100
        self.total_sum = 1000.0
        self.density = 0.1

    def test_generate_balanced_dense(self):
        """Test generation of balanced dense matrices."""
        matrix, row_sums, col_sums = MatrixGenerator.generate_balanced_dense(
            self.rows, self.cols, total_sum=self.total_sum, noise_level=0.05
        )

        # Check dimensions
        assert matrix.shape == (self.rows, self.cols)

        # Check if matrix is balanced
        actual_row_sums = matrix.sum(axis=1)
        actual_col_sums = matrix.sum(axis=0)

        assert np.allclose(actual_row_sums, row_sums, rtol=1e-6)
        assert np.allclose(actual_col_sums, col_sums, rtol=1e-6)
        assert np.allclose(matrix.sum(), self.total_sum, rtol=1e-6)

    def test_generate_balanced_dense_with_gras(self):
        """Test generation of balanced dense matrices using GRAS method."""
        matrix, row_sums, col_sums = MatrixGenerator.generate_balanced_dense(
            self.rows, self.cols, total_sum=self.total_sum, method="GRAS", noise_level=0.2
        )

        # Check dimensions
        assert matrix.shape == (self.rows, self.cols)

        # Check if matrix is balanced
        actual_row_sums = matrix.sum(axis=1)
        actual_col_sums = matrix.sum(axis=0)

        assert np.allclose(actual_row_sums.flatten(), row_sums.flatten(), rtol=1e-6)
        assert np.allclose(actual_col_sums.flatten(), col_sums.flatten(), rtol=1e-6)
        assert np.allclose(matrix.sum(), self.total_sum, rtol=1e-6)

    def test_generate_balanced_sparse(self):
        """Test generation of balanced sparse matrices."""
        matrix, row_sums, col_sums = MatrixGenerator.generate_balanced_sparse(
            self.rows, self.cols, density=self.density, total_sum=self.total_sum
        )

        # Check if result is sparse
        assert sp.issparse(matrix)

        # Check sparsity
        actual_density = matrix.nnz / (self.rows * self.cols)
        assert abs(actual_density - self.density) < 0.01

        # Check balance
        actual_row_sums = matrix.sum(axis=1)
        actual_col_sums = matrix.sum(axis=0)

        assert np.allclose(actual_row_sums.reshape(100, 1).flatten(), row_sums, rtol=1e-6)
        assert np.allclose(actual_col_sums.reshape(100, 1).flatten(), col_sums, rtol=1e-6)
        assert np.allclose(matrix.sum(), self.total_sum, rtol=1e-6)

    def test_generate_balanced_sparse_with_gras(self):
        """Test generation of balanced sparse matrices using RAS method."""
        matrix, row_sums, col_sums = MatrixGenerator.generate_balanced_sparse(
            self.rows, self.cols, density=self.density, total_sum=self.total_sum, method="GRAS"
        )

        # Check if result is sparse
        assert sp.issparse(matrix)

        # Check sparsity
        actual_density = matrix.nnz / (self.rows * self.cols)
        assert abs(actual_density - self.density) < 0.01

        # Check balance
        actual_row_sums = matrix.sum(axis=1)
        actual_col_sums = matrix.sum(axis=0)

        assert np.allclose(actual_row_sums.reshape(100, 1).flatten(), row_sums.flatten(), rtol=1e-6)
        assert np.allclose(actual_col_sums.reshape(100, 1).flatten(), col_sums.flatten(), rtol=1e-6)
        assert np.allclose(matrix.sum(), self.total_sum, rtol=1e-6)

    def test_invalid_density(self):
        """Test handling of invalid density values."""
        with pytest.raises(ValueError):
            MatrixGenerator.generate_balanced_sparse(
                self.rows, self.cols, density=-0.1, total_sum=self.total_sum
            )
        with pytest.raises(ValueError):
            MatrixGenerator.generate_balanced_sparse(
                self.rows, self.cols, density=1.1, total_sum=self.total_sum
            )

    def test_output_type_consistency(self):
        """Test that generated results match the expected structure."""
        dense_result = MatrixGenerator.generate_balanced_dense(
            self.rows, self.cols, total_sum=self.total_sum
        )
        sparse_result = MatrixGenerator.generate_balanced_sparse(
            self.rows, self.cols, density=self.density, total_sum=self.total_sum
        )

        # Verify types
        assert isinstance(dense_result, tuple) and len(dense_result) == 3
        assert isinstance(sparse_result, tuple) and len(sparse_result) == 3

        # Verify data structure for row and column sums
        assert isinstance(dense_result[1], np.ndarray) and isinstance(dense_result[2], np.ndarray)
        assert isinstance(sparse_result[1], np.ndarray) and isinstance(sparse_result[2], np.ndarray)

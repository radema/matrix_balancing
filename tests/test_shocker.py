import numpy as np
import pytest
import scipy.sparse as sp

from ras_balancer import MatrixGenerator, MatrixShocker, ShockType


class TestMatrixShocker:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.rows = 50
        self.cols = 50
        self.total_sum = 1000.0
        self.matrix, self.row_sums, self.col_sums = MatrixGenerator.generate_balanced_dense(
            self.rows, self.cols, total_sum=self.total_sum
        )
        self.sparse_matrix, _, _ = MatrixGenerator.generate_balanced_sparse(
            self.rows, self.cols, density=0.2, total_sum=self.total_sum
        )
        self.shocker = MatrixShocker(preserve_zeros=True, random_seed=42)

    def test_shock_cell(self):
        """Test cell-specific shock."""
        row, col = 0, 0
        magnitude = 0.5
        result = self.shocker.shock_cell(
            self.matrix, row=row, col=col, magnitude=magnitude, relative=True
        )

        # Check shock type
        assert result.shock_type == ShockType.CELL

        # Check if only specified cell was modified
        diff_matrix = result.shocked_matrix - self.matrix
        diff_matrix[row, col] = 0
        assert np.allclose(diff_matrix, 0)

        # Check shock magnitude
        original_value = self.matrix[row, col]
        shocked_value = result.shocked_matrix[row, col]
        assert np.allclose(shocked_value, original_value * (1 + magnitude))

    def test_shock_row_totals(self):
        """Test row total shock."""
        row_indices = [0, 1]
        magnitudes = [0.1, 0.2]
        result = self.shocker.shock_row_totals(
            self.matrix, row_indices=row_indices, magnitudes=magnitudes, relative=True
        )

        # Check shock type
        assert result.shock_type == ShockType.ROW_TOTAL

        # Check if row totals were modified correctly
        for idx, mag in zip(row_indices, magnitudes):
            original_sum = self.matrix[idx].sum()
            shocked_sum = result.shocked_matrix[idx].sum()
            assert np.allclose(shocked_sum, original_sum * (1 + mag))

        # Check if other rows remained unchanged
        mask = np.ones(self.rows, dtype=bool)
        mask[row_indices] = False
        unchanged_rows = result.shocked_matrix[mask] - self.matrix[mask]
        assert np.allclose(unchanged_rows, 0)

    def test_shock_column_totals(self):
        """Test column total shock."""
        col_indices = [0, 1]
        magnitudes = [0.1, 0.2]
        result = self.shocker.shock_column_totals(
            self.matrix, col_indices=col_indices, magnitudes=magnitudes, relative=True
        )

        # Check shock type
        assert result.shock_type == ShockType.COLUMN_TOTAL

        # Check if column totals were modified correctly
        for idx, mag in zip(col_indices, magnitudes):
            original_sum = self.matrix[:, idx].sum()
            shocked_sum = result.shocked_matrix[:, idx].sum()
            assert np.allclose(shocked_sum, original_sum * (1 + mag))

        # Check if other columns remained unchanged
        mask = np.ones(self.cols, dtype=bool)
        mask[col_indices] = False
        unchanged_cols = result.shocked_matrix[:, mask] - self.matrix[:, mask]
        assert np.allclose(unchanged_cols, 0)

    def test_shock_proportional(self):
        """Test proportional shock."""
        magnitude = 0.1
        affected_fraction = 0.05
        result = self.shocker.shock_proportional(
            self.matrix, magnitude=magnitude, affected_fraction=affected_fraction
        )

        # Check shock type
        assert result.shock_type == ShockType.PROPORTIONAL

        # Check number of affected cells
        diff_matrix = result.shocked_matrix - self.matrix
        affected_cells = np.count_nonzero(diff_matrix)
        expected_affected = int(self.matrix.size * affected_fraction)
        assert abs(affected_cells - expected_affected) < expected_affected * 0.1

    def test_shock_sparse_matrix(self):
        """Test shocks on sparse matrices."""
        magnitude = 0.1
        result = self.shocker.shock_proportional(
            self.sparse_matrix, magnitude=magnitude, affected_fraction=0.05
        )

        # Ensure result is sparse
        assert sp.issparse(result.shocked_matrix)

        # Check that shocks are applied
        original_nnz = self.sparse_matrix.nnz
        shocked_nnz = result.shocked_matrix.nnz
        assert shocked_nnz == original_nnz

    def test_preserve_zeros(self):
        """Test zero preservation in shocks."""
        # Create matrix with some zeros
        matrix_with_zeros = self.matrix.copy()
        matrix_with_zeros[0:2, 0:2] = 0

        # Test different shock types
        shock_methods = [
            lambda m: self.shocker.shock_cell(m, 0, 0, 0.5),
            lambda m: self.shocker.shock_row_totals(m, [0], [0.1]),
            lambda m: self.shocker.shock_proportional(m, 0.1, 0.1),
            lambda m: self.shocker.shock_random(m, 0.1, 10),
        ]

        for shock_method in shock_methods:
            result = shock_method(matrix_with_zeros)
            zero_mask = matrix_with_zeros == 0
            assert np.all(result.shocked_matrix[zero_mask] == 0)

    def test_shock_and_rebalance(self):
        """Test shock and rebalance functionality."""
        result = self.shocker.shock_and_rebalance(
            self.matrix, self.row_sums, self.col_sums, method="RAS", max_iter=5000
        )

        # Ensure matrix is rebalanced
        new_row_sums = result.shocked_matrix.sum(axis=1)
        new_col_sums = result.shocked_matrix.sum(axis=0)
        assert np.allclose(new_row_sums, self.row_sums, rtol=1e-6)
        assert np.allclose(new_col_sums, self.col_sums, rtol=1e-6)

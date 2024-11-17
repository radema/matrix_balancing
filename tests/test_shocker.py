import unittest
import numpy as np
import scipy.sparse as sp
from ras_library import (
    RASBalancer, 
    MatrixGenerator, 
    MatrixShocker,
    BalanceStatus,
    ShockType,
    balance_matrix
)

class TestMatrixShocker(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.rows = 50
        self.cols = 50
        self.matrix, self.row_sums, self.col_sums = MatrixGenerator.generate_balanced_dense(
            self.rows,
            self.cols,
            total_sum=1000.0
        )
        self.shocker = MatrixShocker(preserve_zeros=True, random_seed=42)

    def test_shock_cell(self):
        """Test cell-specific shock."""
        row, col = 0, 0
        magnitude = 0.5
        result = self.shocker.shock_cell(
            self.matrix,
            row=row,
            col=col,
            magnitude=magnitude,
            relative=True
        )
        
        # Check shock type
        self.assertEqual(result.shock_type, ShockType.CELL)
        
        # Check if only specified cell was modified
        diff_matrix = result.shocked_matrix - self.matrix
        diff_matrix[row, col] = 0
        self.assertTrue(np.allclose(diff_matrix, 0))
        
        # Check shock magnitude
        original_value = self.matrix[row, col]
        shocked_value = result.shocked_matrix[row, col]
        self.assertTrue(np.allclose(shocked_value, original_value * (1 + magnitude)))

    def test_shock_row_totals(self):
        """Test row total shock."""
        row_indices = [0, 1]
        magnitudes = [0.1, 0.2]
        result = self.shocker.shock_row_totals(
            self.matrix,
            row_indices=row_indices,
            magnitudes=magnitudes,
            relative=True
        )
        
        # Check shock type
        self.assertEqual(result.shock_type, ShockType.ROW_TOTAL)
        
        # Check if row totals were modified correctly
        for idx, mag in zip(row_indices, magnitudes):
            original_sum = self.matrix[idx].sum()
            shocked_sum = result.shocked_matrix[idx].sum()
            self.assertTrue(np.allclose(shocked_sum, original_sum * (1 + mag)))
        
        # Check if other rows remained unchanged
        mask = np.ones(self.rows, dtype=bool)
        mask[row_indices] = False
        unchanged_rows = result.shocked_matrix[mask] - self.matrix[mask]
        self.assertTrue(np.allclose(unchanged_rows, 0))

    def test_shock_proportional(self):
        """Test proportional shock."""
        magnitude = 0.1
        affected_fraction = 0.05
        result = self.shocker.shock_proportional(
            self.matrix,
            magnitude=magnitude,
            affected_fraction=affected_fraction
        )
        
        # Check shock type
        self.assertEqual(result.shock_type, ShockType.PROPORTIONAL)
        
        # Check number of affected cells
        diff_matrix = result.shocked_matrix - self.matrix
        affected_cells = np.count_nonzero(diff_matrix)
        expected_affected = int(self.matrix.size * affected_fraction)
        self.assertLess(abs(affected_cells - expected_affected), expected_affected * 0.1)

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
            lambda m: self.shocker.shock_random(m, 0.1, 10)
        ]
        
        for shock_method in shock_methods:
            result = shock_method(matrix_with_zeros)
            zero_mask = matrix_with_zeros == 0
            self.assertTrue(np.all(result.shocked_matrix[zero_mask] == 0))
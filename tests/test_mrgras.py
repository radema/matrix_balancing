import numpy as np
import pytest
from ras_balancer import MRGRASBalancer, balance_matrix


class TestMRGRASBalancer:
    @pytest.fixture(autouse=True)
    def setup_balancer(self):
        """Set up test fixtures."""
        self.rows = 5
        self.cols = 4
        self.mrgras_balancer = MRGRASBalancer(max_iter=1000, tolerance=1e-8)

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
        self.target_row_sums = np.array([10.0, 10.0, 10.0, 10.0, 10.0]).reshape(-1, 1)
        self.target_col_sums = np.array([12.5, 12.5, 12.5, 12.5]).reshape(-1, 1)
        self.constraints = [
            {(0, 1), (1, 2)},
            {(2, 3), (3, 0)},
        ]
        self.constraint_values = [5.0, 7.0]

    def test_mrgras_balance_with_constraints(self):
        """Test MRGRAS balancing with constraints."""
        result = self.mrgras_balancer.balance(
            matrix=self.test_matrix,
            target_row_sums=self.target_row_sums,
            target_col_sums=self.target_col_sums,
            constraints=self.constraints,
            values=self.constraint_values,
        )

        # Check convergence
        assert result.converged

        # Check row sums
        row_sums = result.balanced_matrix.sum(axis=1)
        assert np.allclose(row_sums, self.target_row_sums, rtol=1e-6)

        # Check column sums
        col_sums = result.balanced_matrix.sum(axis=0)
        assert np.allclose(col_sums, self.target_col_sums, rtol=1e-6)

        G, Q, W = self.mrgras_balancer.construct_constraint_matrices(
            self.constraints, self.constraint_values, *self.test_matrix.shape
        )

        # Check constraint values
        assert np.allclose(
            np.nan_to_num((G @ result.balanced_matrix @ Q - W), nan=1e-10), 0, atol=1e-6
        ), "Constraints sum mismatch."

    def test_mrgras_constraint_validation(self):
        """Test constraint validation in MRGRAS."""
        invalid_constraints = [
            {(0, 0), (self.rows, self.cols)},  # Out-of-bound indices
        ]

        with pytest.raises(IndexError):
            self.mrgras_balancer.balance(
                matrix=self.test_matrix,
                target_row_sums=self.target_row_sums,
                target_col_sums=self.target_col_sums,
                constraints=invalid_constraints,
                values=self.constraint_values,
            )

    def test_balance_matrix_function(self):
        """Test the generic balance_matrix function for MRGRAS."""
        result = balance_matrix(
            matrix=self.test_matrix,
            target_row_sums=self.target_row_sums,
            target_col_sums=self.target_col_sums,
            method="MRGRAS",
            constraints=self.constraints,
            values=self.constraint_values,
        )

        assert result.converged
        assert np.allclose(result.balanced_matrix.sum(axis=1), self.target_row_sums, rtol=1e-6)
        assert np.allclose(result.balanced_matrix.sum(axis=0), self.target_col_sums, rtol=1e-6)

        G, Q, W = self.mrgras_balancer.construct_constraint_matrices(
            self.constraints, self.constraint_values, *self.test_matrix.shape
        )
        # Check constraint values
        assert np.allclose(
            np.nan_to_num((G @ result.balanced_matrix @ Q - W), nan=1e-10), 0, atol=1e-6
        ), "Constraints sum mismatch."

    def test_small_basic_matrix_single_constraint(self):
        """Test small matrix with single cell constraint."""
        X1 = np.array([[5, -3, 0, 1], [2, 1, -2, 1], [-1, 3, 4, 1]])
        u1 = np.array([5, 3, 1]).reshape(-1, 1)
        v1 = np.array([3, 4, 1, 1]).reshape(-1, 1)

        constraints = [{(0, 0)}, {(1, 1)}]
        values = [6, 3]

        result = self.mrgras_balancer.balance(
            matrix=X1,
            target_row_sums=u1,
            target_col_sums=v1,
            constraints=constraints,
            values=values,
        )

        assert result.converged
        assert np.allclose(result.balanced_matrix.sum(axis=1), u1.flatten(), atol=1e-10)
        assert np.allclose(result.balanced_matrix.sum(axis=0), v1.flatten(), atol=1e-10)

        # Check constraint values
        for i, constraint_set in enumerate(constraints):
            constraint_sum = sum(result.balanced_matrix[row, col] for row, col in constraint_set)
            assert np.isclose(
                constraint_sum, values[i], atol=1e-10
            ), f"Constraint {i} sum mismatch. Expected {values[i]}, got {constraint_sum}."

    def test_diagonal_matrix(self):
        """Test diagonal matrix with constraints."""
        X2 = np.array([[5, 0, 0], [0, -4, 0], [0, 0, 2]])
        u2 = np.array([5, -4, 3]).reshape(-1, 1)
        v2 = np.array([5, -4, 3]).reshape(-1, 1)

        constraints = [{(0, 0)}, {(1, 1)}]
        values = [5.0, -4.0]

        result = self.mrgras_balancer.balance(
            matrix=X2,
            target_row_sums=u2,
            target_col_sums=v2,
            constraints=constraints,
            values=values,
        )

        assert result.converged
        assert np.allclose(result.balanced_matrix.sum(axis=1), u2.flatten(), atol=1e-10)
        assert np.allclose(result.balanced_matrix.sum(axis=0), v2.flatten(), atol=1e-10)

        # Check constraint values
        for i, constraint_set in enumerate(constraints):
            constraint_sum = sum(result.balanced_matrix[row, col] for row, col in constraint_set)
            assert np.isclose(
                constraint_sum, values[i], atol=1e-10
            ), f"Constraint {i} sum mismatch. Expected {values[i]}, got {constraint_sum}."

    def test_all_positive_matrix(self):
        """Test all positive matrix with constraints."""
        X3 = np.array([[3, 2, 0], [4, 5, 0], [0, 0, 9]])
        u3 = np.array([9, 16, 11]).reshape(-1, 1)
        v3 = np.array([5, 20, 11]).reshape(-1, 1)

        constraints = [{(0, 0), (0, 1), (1, 0), (1, 1)}, {(2, 2)}]
        values = [25, 11]

        result = self.mrgras_balancer.balance(
            matrix=X3,
            target_row_sums=u3,
            target_col_sums=v3,
            constraints=constraints,
            values=values,
        )

        assert result.converged
        assert np.allclose(result.balanced_matrix.sum(axis=1), u3.flatten(), atol=1e-10)
        assert np.allclose(result.balanced_matrix.sum(axis=0), v3.flatten(), atol=1e-10)

        # Check constraint values
        for i, constraint_set in enumerate(constraints):
            constraint_sum = sum(result.balanced_matrix[row, col] for row, col in constraint_set)
            assert np.isclose(
                constraint_sum, values[i], atol=1e-10
            ), f"Constraint {i} sum mismatch. Expected {values[i]}, got {constraint_sum}."

    def test_large_random_matrix(self):
        """Test a large random matrix with constraints."""
        N = 1000
        X_test = np.random.rand(N, N)

        u = np.ones(N).reshape(-1, 1) * 4
        v = np.ones(N).reshape(-1, 1) * 4

        constraints = [{(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)}]
        values = [9]

        result = self.mrgras_balancer.balance(
            matrix=X_test,
            target_row_sums=u,
            target_col_sums=v,
            constraints=constraints,
            values=values,
            max_iter=2000,
        )

        assert result.converged

        # Check constraint values
        for i, constraint_set in enumerate(constraints):
            constraint_sum = sum(result.balanced_matrix[row, col] for row, col in constraint_set)
            assert np.isclose(
                constraint_sum, values[i], atol=1e-10
            ), f"Constraint {i} sum mismatch. Expected {values[i]}, got {constraint_sum}."

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.typing import NDArray

from ras_balancer.operations import (
    assign_row,
    count_elements,
    extract_row,
    get_nonzero_indices,
    safe_transpose,
)


class TestOperations:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.dense_matrix = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
        self.sparse_matrix = sp.csr_matrix(self.dense_matrix)

    def test_safe_transpose_dense(self):
        """Test safe_transpose with dense matrix."""
        transposed = safe_transpose(self.dense_matrix)
        assert np.array_equal(transposed, self.dense_matrix.T)
        assert not sp.issparse(transposed)

    def test_safe_transpose_sparse(self):
        """Test safe_transpose with sparse matrix."""
        transposed = safe_transpose(self.sparse_matrix)
        assert sp.issparse(transposed)
        assert np.array_equal(transposed.toarray(), self.dense_matrix.T)

    def test_count_elements_dense_preserve_zeros(self):
        """Test count_elements with dense matrix and preserve_zeros=True."""
        count = count_elements(self.dense_matrix, preserve_zeros=True)
        assert count == np.count_nonzero(self.dense_matrix)
        assert count == 6

    def test_count_elements_dense_no_preserve_zeros(self):
        """Test count_elements with dense matrix and preserve_zeros=False."""
        count = count_elements(self.dense_matrix, preserve_zeros=False)
        assert count == self.dense_matrix.size
        assert count == 9

    def test_count_elements_sparse(self):
        """Test count_elements with sparse matrix."""
        # For sparse matrices, it should typically return nnz
        count = count_elements(self.sparse_matrix, preserve_zeros=True)
        assert count == self.sparse_matrix.nnz
        assert count == 6

        # Even if preserve_zeros=False, sparse matrices might be treated as nnz
        # based on existing logic in shocker.py:
        # if sp.issparse(matrix): n_elements = matrix.nnz
        count = count_elements(self.sparse_matrix, preserve_zeros=False)
        assert count == self.sparse_matrix.nnz

    def test_get_nonzero_indices_dense(self):
        """Test get_nonzero_indices with dense matrix."""
        rows, cols = get_nonzero_indices(self.dense_matrix)
        expected_rows, expected_cols = np.where(self.dense_matrix != 0)
        assert np.array_equal(rows, expected_rows)
        assert np.array_equal(cols, expected_cols)

    def test_get_nonzero_indices_sparse(self):
        """Test get_nonzero_indices with sparse matrix."""
        rows, cols = get_nonzero_indices(self.sparse_matrix)
        expected_rows, expected_cols = self.sparse_matrix.nonzero()
        assert np.array_equal(rows, expected_rows)
        assert np.array_equal(cols, expected_cols)

    def test_extract_row_dense(self):
        """Test extract_row with dense matrix."""
        row_idx = 1
        row_values = extract_row(self.dense_matrix, row_idx)
        assert np.array_equal(row_values, self.dense_matrix[row_idx])

    def test_extract_row_sparse(self):
        """Test extract_row with sparse matrix."""
        row_idx = 1
        row_values = extract_row(self.sparse_matrix, row_idx)
        # Should return a dense array (ravelled)
        assert isinstance(row_values, np.ndarray)
        assert row_values.ndim == 1
        assert np.array_equal(row_values, self.dense_matrix[row_idx])

    def test_assign_row_dense(self):
        """Test assign_row with dense matrix."""
        matrix = self.dense_matrix.copy()
        row_idx = 1
        new_values = np.array([7, 8, 9])
        assign_row(matrix, row_idx, new_values)
        assert np.array_equal(matrix[row_idx], new_values)

    def test_assign_row_sparse(self):
        """Test assign_row with sparse matrix."""
        matrix = self.sparse_matrix.copy()
        row_idx = 1
        new_values = np.array([7, 8, 9])
        assign_row(matrix, row_idx, new_values)
        assert sp.issparse(matrix)
        assert np.array_equal(matrix[row_idx].toarray().ravel(), new_values)

    def test_empty_matrix(self):
        """Test operations on empty/small matrix."""
        empty_dense = np.zeros((2, 2))
        empty_sparse = sp.csr_matrix(empty_dense)

        assert count_elements(empty_dense, True) == 0
        assert count_elements(empty_dense, False) == 4

        rows, cols = get_nonzero_indices(empty_dense)
        assert len(rows) == 0
        assert len(cols) == 0

        rows, cols = get_nonzero_indices(empty_sparse)
        assert len(rows) == 0
        assert len(cols) == 0

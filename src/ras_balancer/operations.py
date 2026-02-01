"""Helper operations for matrix manipulation handling both sparse and dense matrices."""

from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


def safe_transpose(matrix: Union[NDArray, sp.spmatrix]) -> Union[NDArray, sp.spmatrix]:
    """Safely transpose matrix, handling both sparse and dense types."""
    return matrix.transpose() if sp.issparse(matrix) else matrix.T


def count_elements(matrix: Union[NDArray, sp.spmatrix], preserve_zeros: bool) -> int:
    """Count elements according to matrix type and preserve_zeros setting."""
    if sp.issparse(matrix):
        return matrix.nnz
    elif preserve_zeros:
        return np.count_nonzero(matrix)
    else:
        return matrix.size


def get_nonzero_indices(matrix: Union[NDArray, sp.spmatrix]) -> Tuple[NDArray, NDArray]:
    """Get indices of nonzero elements for both sparse and dense matrices."""
    if sp.issparse(matrix):
        return matrix.nonzero()
    else:
        return np.where(matrix != 0)


def extract_row(matrix: Union[NDArray, sp.spmatrix], row_index: int) -> NDArray:
    """Extract row values, handling both sparse and dense matrices.

    Returns a dense 1D array.
    """
    row = matrix[row_index]
    return row.toarray().ravel() if sp.issparse(matrix) else row


def assign_row(matrix: Union[NDArray, sp.spmatrix], row_index: int, values: NDArray) -> None:
    """Assign values to a row, handling both sparse and dense matrices."""
    if sp.issparse(matrix):
        matrix[row_index] = sp.csr_matrix(values)
    else:
        matrix[row_index] = values

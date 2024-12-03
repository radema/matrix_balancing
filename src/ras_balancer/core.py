# core.py
"""Core functionality for matrix balancing using the RAS algorithm."""

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
import logging
import warnings
from typing import Union, Optional
from .types import BalanceStatus, BalanceCheckResult, RASResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatrixBalancerBase:
    """Base class for matrix balancing algorithms."""
    def __init__(
        self,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        use_sparse: Optional[bool] = None,
        chunk_size: int = 1000,
    ):
        """Initialize the RAS Balancer."""
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.use_sparse = use_sparse
        self.chunk_size = chunk_size

    def _validate_inputs(self, matrix, target_row_sums, target_col_sums):
        if matrix.shape[0] != len(target_row_sums) or matrix.shape[1] != len(target_col_sums):
            raise ValueError("Target sum dimensions must match matrix dimensions")
        if np.any(target_row_sums < 0) or np.any(target_col_sums < 0):
            raise ValueError("Target sums must be non-negative")
        if not np.allclose(target_row_sums.sum(), target_col_sums.sum(), rtol=1e-5):
            raise ValueError("Sum of target row sums must equal sum of target column sums")

    def _should_use_sparse(self, matrix: Union[NDArray, sp.spmatrix]) -> bool:
        """Determine if sparse matrix operations should be used."""
        if self.use_sparse is not None:
            return self.use_sparse

        if sp.issparse(matrix):
            return True

        # For dense matrices, estimate sparsity
        if isinstance(matrix, np.ndarray):
            sparsity = np.count_nonzero(matrix) / matrix.size
            return sparsity < 0.1

        return False
    
    @staticmethod
    def check_balance(
        matrix: Union[NDArray, sp.spmatrix],
        target_row_sums: Optional[NDArray] = None,
        target_col_sums: Optional[NDArray] = None,
        tolerance: float = 1e-6,
    ) -> BalanceCheckResult:
        """Check if a matrix is balanced according to target row and column sums."""
        current_row_sums = matrix.sum(axis=1).A1 if sp.issparse(matrix) else matrix.sum(axis=1)
        current_col_sums = matrix.sum(axis=0).A1 if sp.issparse(matrix) else matrix.sum(axis=0)

        target_row_sums = current_row_sums if target_row_sums is None else target_row_sums
        target_col_sums = current_col_sums if target_col_sums is None else target_col_sums

        row_deviations = np.abs(current_row_sums - target_row_sums) / np.maximum(
            target_row_sums, 1e-10
        )
        col_deviations = np.abs(current_col_sums - target_col_sums) / np.maximum(
            target_col_sums, 1e-10
        )

        max_row_dev = np.max(row_deviations)
        max_col_dev = np.max(col_deviations)

        row_balanced = max_row_dev <= tolerance
        col_balanced = max_col_dev <= tolerance

        if row_balanced and col_balanced:
            status = BalanceStatus.BALANCED
        elif row_balanced:
            status = BalanceStatus.UNBALANCED_COLS
        elif col_balanced:
            status = BalanceStatus.UNBALANCED_ROWS
        else:
            status = BalanceStatus.UNBALANCED_BOTH

        return BalanceCheckResult(
            status=status,
            max_row_deviation=max_row_dev,
            max_col_deviation=max_col_dev,
            row_deviations=row_deviations,
            col_deviations=col_deviations,
            is_balanced=row_balanced and col_balanced,
        )
    
    def process_dense_chunk(
            self, matrix: NDArray, r: NDArray, s: NDArray, start: int, end: int
            ) -> NDArray:
        """Process a chunk of a dense matrix for memory efficiency."""
        chunk = matrix[start:end, :]
        chunk = np.multiply(chunk, r[start:end, np.newaxis])
        chunk = np.multiply(chunk, s)
        return chunk
    
class RASBalancer(MatrixBalancerBase):
    """A class for balancing matrices using the RAS algorithm."""

    def balance(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        target_row_sums: NDArray,
        target_col_sums: NDArray,
        initial_correction: bool = True,
    ) -> RASResult:
        """
        Balance a matrix using the RAS algorithm.

        Parameters
        ----------
        matrix : Union[NDArray, sp.spmatrix]
            Input matrix to balance
        target_row_sums : NDArray
            Target row sums
        target_col_sums : NDArray
            Target column sums
        initial_correction : bool, optional
            Apply initial scaling correction, by default True

        Returns
        -------
        RASResult
            Results containing balanced matrix and convergence information
        """
        self._validate_inputs(matrix, target_row_sums, target_col_sums)
        use_sparse = self._should_use_sparse(matrix)

        # Convert to sparse if needed
        if use_sparse and not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        elif not use_sparse and sp.issparse(matrix):
            matrix = matrix.toarray()

        # Work with copies
        X = matrix.copy()
        target_row_sums = np.asarray(target_row_sums, dtype=np.float64).flatten()
        target_col_sums = np.asarray(target_col_sums, dtype=np.float64).flatten()

        # Initial correction to improve convergence
        if initial_correction:
            total_sum = target_row_sums.sum()
            X = X * (total_sum / X.sum())

        for iteration in range(self.max_iter):
            # Row scaling
            row_sums = X.sum(axis=1).A1 if use_sparse else X.sum(axis=1)
            r = np.divide(target_row_sums, row_sums, where=row_sums != 0)

            if use_sparse:
                X = sp.diags(r) @ X
            else:
                # Process large dense matrices in chunks
                if X.shape[0] > self.chunk_size:
                    new_X = np.empty_like(X)
                    for i in range(0, X.shape[0], self.chunk_size):
                        end = min(i + self.chunk_size, X.shape[0])
                        new_X[i:end] = self._process_dense_chunk(X, r, np.ones(X.shape[1]), i, end)
                    X = new_X
                else:
                    X = np.multiply(X, r[:, np.newaxis])

            # Column scaling
            col_sums = X.sum(axis=0).A1 if use_sparse else X.sum(axis=0)
            s = np.divide(target_col_sums, col_sums, where=col_sums != 0)

            if use_sparse:
                X = X @ sp.diags(s)
            else:
                X = np.multiply(X, s)

            # Check convergence
            current_row_sums = X.sum(axis=1).A1 if use_sparse else X.sum(axis=1)
            current_col_sums = X.sum(axis=0).A1 if use_sparse else X.sum(axis=0)

            row_error = np.max(np.abs(current_row_sums - target_row_sums))
            col_error = np.max(np.abs(current_col_sums - target_col_sums))

            if max(row_error, col_error) < self.tolerance:
                return RASResult(X, iteration + 1, True, row_error, col_error)

        warnings.warn("RAS algorithm did not converge within maximum iterations")
        return RASResult(X, self.max_iter, False, row_error, col_error)
    

class GRASBalancer(MatrixBalancerBase):
    """Balances matrices using the GRAS algorithm for matrices with both positive and negative values."""

    @staticmethod
    def invd_sparse(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            invd_values = np.where(x != 0, 1.0 / x, 1.0)
        return sp.diags(invd_values.flatten())

    def balance(self, matrix, target_row_sums, target_col_sums):
        self._validate_inputs(matrix, target_row_sums, target_col_sums)

        m,n = matrix.shape

        if sp.issparse(matrix):
            # For sparse matrices, use `.maximum` for efficient operations
            P = matrix.maximum(0)  # Extract positive part
            N = matrix.minimum(0).multiply(-1)  # Extract negative part and negate
        else:
            # For dense numpy arrays, use `np.maximum` directly
            P = np.maximum(matrix, 0)  # Extract positive part
            N = np.maximum(-matrix, 0)  # Extract negative part and negate

        r = np.ones((m, 1))
        dif = float('inf')

        pr = P.T @ r
        nr = N.T @ self.invd_sparse(r) @ np.ones((m, 1))

        s = self.invd_sparse(2 * pr) @ (target_col_sums.reshape(-1,1) + np.sqrt(target_col_sums.reshape(-1,1)**2 + 4 * pr * nr))
        # Handle possible NaNs
        s = np.nan_to_num(s, nan=1e-10, posinf=1e-10, neginf=1e-10)
        ss = -self.invd_sparse(target_col_sums.reshape(-1,1)) @ nr
        s[pr.flatten() == 0] = ss[pr.flatten() == 0]

        s_old = s
        r_old = r

        for iteration in range(self.max_iter):
            # Update row and column scaling factors

            ps = P @ s
            ns = N @ self.invd_sparse(s) @ np.ones((n, 1))
            r = self.invd_sparse(2 * ps) @ (target_row_sums.reshape(-1,1) + np.sqrt(target_row_sums.reshape(-1,1)**2 + 4 * ps * ns))
            # Handle possible NaNs
            r = np.nan_to_num(r, nan=1e-10, posinf=1e-10, neginf=1e-10)            
            rr = -self.invd_sparse(target_row_sums.reshape(-1,1)) @ ns
            r[ps.flatten() == 0] = rr[ps.flatten() == 0]

            pr = P.T @ r
            nr = N.T @ self.invd_sparse(r) @ np.ones((m, 1))

            s = self.invd_sparse(2 * pr) @ (target_col_sums.reshape(-1,1) + np.sqrt(target_col_sums.reshape(-1,1)**2 + 4 * pr * nr))
            # Handle possible NaNs
            s = np.nan_to_num(s, nan=1e-10, posinf=1e-10, neginf=1e-10)
            ss = -self.invd_sparse(target_col_sums.reshape(-1,1)) @ nr
            s[pr.flatten() == 0] = ss[pr.flatten() == 0]

            s_dif = np.max(np.abs(s - s_old))
            r_dif = np.max(np.abs(r - r_old))

            dif = max(s_dif, r_dif)

            if (dif < self.tolerance):
                break

            
            s_old = s
            r_old = r

        if dif > self.tolerance:
            warnings.warn("GRAS algorithm did not converge within the maximum number of iterations")
            return RASResult(matrix, self.max_iter, False, dif, dif)

        balanced_matrix = sp.diags(r.flatten()) @ P @ sp.diags(s.flatten()) - self.invd_sparse(r) @ N @ self.invd_sparse(s)
        return RASResult(balanced_matrix, iteration + 1, True, dif, dif)

def balance_matrix(
    matrix: Union[np.ndarray, sp.spmatrix],
    target_row_sums: np.ndarray,
    target_col_sums: np.ndarray,
    method: str = "RAS",
    **kwargs,
) -> RASResult:
    """
    Balances a matrix using the specified method (RAS or GRAS).

    Parameters
    ----------
    matrix : Union[np.ndarray, sp.spmatrix]
        The input matrix to balance.
    target_row_sums : np.ndarray
        Target row sums.
    target_col_sums : np.ndarray
        Target column sums.
    method : str, optional
        The balancing method to use ('RAS' or 'GRAS'), by default 'RAS'.
    **kwargs
        Additional parameters passed to the balancer class (e.g., max_iter, tolerance).

    Returns
    -------
    RASResult
        The result of the balancing process.
    """
    if method.upper() == "RAS":
        balancer = RASBalancer(**kwargs)
    elif method.upper() == "GRAS":
        balancer = GRASBalancer(**kwargs)
    else:
        raise ValueError(f"Unknown balancing method: {method}. Supported methods are 'RAS' and 'GRAS'.")

    return balancer.balance(matrix, target_row_sums, target_col_sums)

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

class RASBalancer:
    """A class for balancing matrices using the RAS algorithm."""
    
    def __init__(
        self,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        use_sparse: Optional[bool] = None,
        chunk_size: int = 1000
    ):
        """Initialize the RAS Balancer."""
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.use_sparse = use_sparse
        self.chunk_size = chunk_size
    
    @staticmethod
    def check_balance(
        matrix: Union[NDArray, sp.spmatrix],
        target_row_sums: Optional[NDArray] = None,
        target_col_sums: Optional[NDArray] = None,
        tolerance: float = 1e-6
    ) -> BalanceCheckResult:
        """Check if a matrix is balanced according to target row and column sums."""
        current_row_sums = matrix.sum(axis=1).A1 if sp.issparse(matrix) else matrix.sum(axis=1)
        current_col_sums = matrix.sum(axis=0).A1 if sp.issparse(matrix) else matrix.sum(axis=0)
        
        target_row_sums = current_row_sums if target_row_sums is None else target_row_sums
        target_col_sums = current_col_sums if target_col_sums is None else target_col_sums
        
        row_deviations = np.abs(current_row_sums - target_row_sums) / np.maximum(target_row_sums, 1e-10)
        col_deviations = np.abs(current_col_sums - target_col_sums) / np.maximum(target_col_sums, 1e-10)
        
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
            is_balanced=row_balanced and col_balanced
        )

    # [Previous methods from RASBalancer would go here]
    # _validate_inputs, _should_use_sparse, _process_dense_chunk, balance
    def _validate_inputs(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        target_row_sums: NDArray,
        target_col_sums: NDArray
    ) -> None:
        """Validate input dimensions and values."""
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

    def _process_dense_chunk(
        self,
        matrix: NDArray,
        r: NDArray,
        s: NDArray,
        start: int,
        end: int
    ) -> NDArray:
        """Process a chunk of a dense matrix for memory efficiency."""
        chunk = matrix[start:end, :]
        chunk = np.multiply(chunk, r[start:end, np.newaxis])
        chunk = np.multiply(chunk, s)
        return chunk

    def balance(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        target_row_sums: NDArray,
        target_col_sums: NDArray,
        initial_correction: bool = True
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
        target_row_sums = np.asarray(target_row_sums, dtype=np.float64)
        target_col_sums = np.asarray(target_col_sums, dtype=np.float64)
        
        # Initial correction to improve convergence
        if initial_correction:
            total_sum = target_row_sums.sum()
            X = X * (total_sum / X.sum())
            
        for iteration in range(self.max_iter):
            # Row scaling
            row_sums = X.sum(axis=1).A1 if use_sparse else X.sum(axis=1)
            r = np.divide(target_row_sums, row_sums, where=row_sums!=0)
            
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
            s = np.divide(target_col_sums, col_sums, where=col_sums!=0)
            
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

def balance_matrix(
    matrix: Union[NDArray, sp.spmatrix],
    target_row_sums: NDArray,
    target_col_sums: NDArray,
    **kwargs
) -> RASResult:
    """Convenience function to balance a matrix using default parameters."""
    balancer = RASBalancer(**kwargs)
    return balancer.balance(matrix, target_row_sums, target_col_sums)

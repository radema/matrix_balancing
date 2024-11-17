# core.py
"""Core functionality for matrix balancing using the RAS algorithm."""

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
import logging
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

def balance_matrix(
    matrix: Union[NDArray, sp.spmatrix],
    target_row_sums: NDArray,
    target_col_sums: NDArray,
    **kwargs
) -> RASResult:
    """Convenience function to balance a matrix using default parameters."""
    balancer = RASBalancer(**kwargs)
    return balancer.balance(matrix, target_row_sums, target_col_sums)

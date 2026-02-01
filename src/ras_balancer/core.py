# core.py
"""Core functionality for matrix balancing using the RAS algorithm."""

import logging
import warnings
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .types import BalanceCheckResult, BalanceStatus, RASResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatrixBalancerBase:
    """Base class for matrix balancing algorithms."""

    EPSILON: float = 1e-10

    def __init__(
        self,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        use_sparse: Optional[bool] = None,
        chunk_size: int = 1000,
    ):
        """Initialize the matrix balancer."""
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.use_sparse = use_sparse
        self.chunk_size = chunk_size

    def _validate_inputs(self, matrix, target_row_sums, target_col_sums):
        if matrix.shape[0] != len(target_row_sums) or matrix.shape[1] != len(target_col_sums):
            raise ValueError("Target sum dimensions must match matrix dimensions")
        # if np.any(target_row_sums < 0) or np.any(target_col_sums < 0):
        #    raise ValueError("Target sums must be non-negative")
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
            target_row_sums, MatrixBalancerBase.EPSILON
        )
        col_deviations = np.abs(current_col_sums - target_col_sums) / np.maximum(
            target_col_sums, MatrixBalancerBase.EPSILON
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
        """
        Process a chunk of a dense matrix for memory efficiency.

        Parameters
        ----------
        matrix : NDArray
            The full dense matrix.
        r : NDArray
            Row scaling factors.
        s : NDArray
            Column scaling factors.
        start : int
            Start index of the chunk.
        end : int
            End index of the chunk.

        Returns
        -------
        NDArray
            The processed chunk of the matrix.
        """
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
        **kwargs,
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
                return RASResult(X, iteration + 1, True, row_error, col_error, r, s)

        warnings.warn("RAS algorithm did not converge within maximum iterations")
        return RASResult(X, self.max_iter, False, row_error, col_error, r, s)


class GRASBalancer(MatrixBalancerBase):
    """
    Balances matrices using the GRAS algorithm
    for matrices with both positive and negative values.
    """

    @staticmethod
    def invd_sparse(x: NDArray) -> sp.diags:
        """
        Compute the inverse of a sparse diagonal matrix.

        Parameters
        ----------
        x : NDArray
            The input array (diagonal elements).

        Returns
        -------
        sp.diags
            The sparse diagonal matrix with inverted values.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            invd_values = np.where(x != 0, 1.0 / x, 1.0)
        return sp.diags(invd_values.flatten())

    def _prepare_gras_matrices(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        bias_matrix: Optional[Union[NDArray, sp.spmatrix]] = None,
        bias_method: str = "multiplicative",
    ) -> Tuple[Union[NDArray, sp.spmatrix], Union[NDArray, sp.spmatrix]]:
        """Prepare positive and negative parts of the matrix for GRAS algorithm."""
        # Check bias matrix if provided
        if bias_matrix is not None:
            # Validate bias matrix dimensions
            if bias_matrix.shape != matrix.shape:
                raise ValueError("Bias matrix must have the same shape as input matrix")

            # Apply bias matrix based on specified method
            if bias_method.lower() == "multiplicative":
                matrix = matrix * bias_matrix
            elif bias_method.lower() == "additive":
                matrix = matrix + bias_matrix
            else:
                raise ValueError("Bias method must be 'multiplicative' or 'additive'")

        if sp.issparse(matrix):
            # For sparse matrices, use `.maximum` for efficient operations
            P = matrix.maximum(0)  # Extract positive part
            N = matrix.minimum(0).multiply(-1)  # Extract negative part and negate
        else:
            # For dense numpy arrays, use `np.maximum` directly
            P = np.maximum(matrix, 0)  # Extract positive part
            N = np.maximum(-matrix, 0)  # Extract negative part and negate

        return P, N

    def _gras_iteration(
        self,
        P: Union[NDArray, sp.spmatrix],
        N: Union[NDArray, sp.spmatrix],
        target_row_sums: NDArray,
        target_col_sums: NDArray,
    ) -> Tuple[NDArray, NDArray, float, int]:
        """Run the main GRAS algorithm iteration loop."""
        m, n = P.shape
        r = np.ones((m, 1))

        # Initial calculation for s
        pr = P.T @ r
        nr = N.T @ self.invd_sparse(r) @ np.ones((m, 1))

        s = self.invd_sparse(2 * pr) @ (
            target_col_sums.reshape(-1, 1)
            + np.sqrt(target_col_sums.reshape(-1, 1) ** 2 + 4 * pr * nr)
        )
        # Handle possible NaNs
        s = np.nan_to_num(s, nan=self.EPSILON, posinf=self.EPSILON, neginf=self.EPSILON)
        ss = -self.invd_sparse(target_col_sums.reshape(-1, 1)) @ nr
        s[pr.flatten() == 0] = ss[pr.flatten() == 0]

        s_old = s
        r_old = r
        dif = float("inf")
        iteration = 0

        for iteration in range(self.max_iter):
            # Update row and column scaling factors

            ps = P @ s
            ns = N @ self.invd_sparse(s) @ np.ones((n, 1))
            r = self.invd_sparse(2 * ps) @ (
                target_row_sums.reshape(-1, 1)
                + np.sqrt(target_row_sums.reshape(-1, 1) ** 2 + 4 * ps * ns)
            )
            # Handle possible NaNs
            r = np.nan_to_num(r, nan=self.EPSILON, posinf=self.EPSILON, neginf=self.EPSILON)
            rr = -self.invd_sparse(target_row_sums.reshape(-1, 1)) @ ns
            r[ps.flatten() == 0] = rr[ps.flatten() == 0]

            pr = P.T @ r
            nr = N.T @ self.invd_sparse(r) @ np.ones((m, 1))

            s = self.invd_sparse(2 * pr) @ (
                target_col_sums.reshape(-1, 1)
                + np.sqrt(target_col_sums.reshape(-1, 1) ** 2 + 4 * pr * nr)
            )
            # Handle possible NaNs
            s = np.nan_to_num(s, nan=self.EPSILON, posinf=self.EPSILON, neginf=self.EPSILON)
            ss = -self.invd_sparse(target_col_sums.reshape(-1, 1)) @ nr
            s[pr.flatten() == 0] = ss[pr.flatten() == 0]

            s_dif = np.max(np.abs(s - s_old))
            r_dif = np.max(np.abs(r - r_old))

            dif = max(s_dif, r_dif)

            if dif < self.tolerance:
                break

            s_old = s
            r_old = r

        return r, s, dif, iteration + 1

    def _build_gras_result(
        self,
        P: Union[NDArray, sp.spmatrix],
        N: Union[NDArray, sp.spmatrix],
        r: NDArray,
        s: NDArray,
        dif: float,
        iteration: int,
        converged: bool,
    ) -> RASResult:
        """Construct the result object from GRAS algorithm outputs."""
        if not converged:
            warnings.warn("GRAS algorithm did not converge within the maximum number of iterations")
            balanced_matrix = P - N
            return RASResult(balanced_matrix, self.max_iter, False, dif, dif, r, s)

        balanced_matrix = sp.diags(r.flatten()) @ P @ sp.diags(s.flatten()) - self.invd_sparse(
            r
        ) @ N @ self.invd_sparse(s)
        return RASResult(balanced_matrix, iteration, True, dif, dif, r, s)

    def balance(
        self,
        matrix: Union[NDArray, sp.spmatrix],
        target_row_sums: NDArray,
        target_col_sums: NDArray,
        bias_matrix: Optional[Union[NDArray, sp.spmatrix]] = None,
        bias_method: str = "multiplicative",
        **kwargs,
    ) -> RASResult:
        """
        Balance a matrix using the GRAS algorithm with optional bias matrix.

        Parameters
        ----------
        matrix : Union[NDArray, sp.spmatrix]
            Input matrix to balance
        target_row_sums : NDArray
            Target row sums
        target_col_sums : NDArray
            Target column sums
        bias_matrix : Optional[Union[NDArray, sp.spmatrix]], optional
            Bias matrix to modify the input matrix before balancing, by default None
        bias_method : str, optional
            Method of applying bias matrix ('multiplicative' or 'additive'),
            by default 'multiplicative'

        Returns
        -------
        RASResult
            Results containing balanced matrix and convergence information
        """
        # Validate inputs
        self._validate_inputs(matrix, target_row_sums, target_col_sums)

        P, N = self._prepare_gras_matrices(matrix, bias_matrix, bias_method)

        r, s, dif, iteration = self._gras_iteration(P, N, target_row_sums, target_col_sums)

        converged = dif < self.tolerance

        return self._build_gras_result(P, N, r, s, dif, iteration, converged)


class MRGRASBalancer(MatrixBalancerBase):
    """Balances matrices using the MRGRAS algorithm for multi-region constraints."""

    @staticmethod
    def invd_sparse(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            invd_values = np.where(x != 0, 1.0 / x, 1.0)
        return sp.diags(invd_values.flatten())

    def construct_constraint_matrices(
        self,
        constraints: List[Set[Tuple[int, int]]],
        values: List[float],
        m: int,  # Number of input rows
        n: int,  # Number of input columns
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct Q, G, and W matrices for multiregion constraints.
        """

        def find_groups(components):
            groups = []
            for comp_set in components:
                matching_groups = [group for group in groups if not group.isdisjoint(comp_set)]
                if matching_groups:
                    merged_group = set.union(*matching_groups, comp_set)
                    groups = [group for group in groups if group not in matching_groups]
                    groups.append(merged_group)
                else:
                    groups.append(comp_set)
            return groups

        first_components = [
            (set(idx[0] for idx in constraint_set)) for constraint_set in constraints
        ]
        second_components = [
            (set(idx[1] for idx in constraint_set)) for constraint_set in constraints
        ]
        row_groups = find_groups(first_components)
        col_groups = find_groups(second_components)

        covered_rows = {i for group in row_groups for i in group}
        covered_cols = {j for group in col_groups for j in group}

        if len(covered_rows) < m:
            uncovered_rows = set(range(m)) - covered_rows
            row_groups.append(uncovered_rows)

        if len(covered_cols) < n:
            uncovered_cols = set(range(n)) - covered_cols
            col_groups.append(uncovered_cols)

        M = len(row_groups)
        N = len(col_groups)

        G = np.zeros((M, m), dtype=float)
        Q = np.zeros((n, N), dtype=float)
        W = np.full((M, N), np.nan, dtype=float)

        row_group_map = {}
        for group_idx, group in enumerate(row_groups):
            for row in group:
                row_group_map[row] = group_idx

        col_group_map = {}
        for group_idx, group in enumerate(col_groups):
            for col in group:
                col_group_map[col] = group_idx

        for row, group_idx in row_group_map.items():
            G[group_idx, row] = 1.0

        for col, group_idx in col_group_map.items():
            Q[col, group_idx] = 1.0

        for constraint_set, value in zip(constraints, values):
            for i, j in constraint_set:
                row_idx = row_group_map[i]
                col_idx = col_group_map[j]
                W[row_idx, col_idx] = value

        return G, Q, W

    def MRGRAS_UpdateMatrix(self, P, N, T, r, s):
        """
        Updates the matrix using the MRGRAS balancing approach with sparse matrices.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            invd_T = np.where(T != 0, 1.0 / T, 1.0)
        return sp.diags(r.flatten()) @ (P * T) @ sp.diags(s.flatten()) - sp.diags(
            1.0 / r.flatten()
        ) @ (N * invd_T) @ sp.diags(1.0 / s.flatten())

    def balance(
        self,
        matrix: Union[np.ndarray, sp.spmatrix],
        target_row_sums: np.ndarray,
        target_col_sums: np.ndarray,
        constraints: List[Set[Tuple[int, int]]] = None,
        values: List[float] = None,
        epsilon_constr: float = 1e-10,
        **kwargs,
    ) -> RASResult:
        """
        Balance a matrix using the MRGRAS algorithm.
        """
        self._validate_inputs(matrix, target_row_sums, target_col_sums)
        use_sparse = self._should_use_sparse(matrix)

        if use_sparse and not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        elif not use_sparse and sp.issparse(matrix):
            matrix = matrix.toarray()

        u = target_row_sums.reshape(-1, 1)
        v = target_col_sums.reshape(-1, 1)

        if constraints and values:
            G, Q, W = self.construct_constraint_matrices(constraints, values, *matrix.shape)
        else:
            G = np.eye(matrix.shape[0])
            Q = np.eye(matrix.shape[1])
            W = np.ones(matrix.shape)

        if sp.issparse(matrix):
            P = matrix.maximum(0)
            N = matrix.minimum(0).multiply(-1)
        else:
            P = np.maximum(matrix, 0)
            N = np.maximum(-matrix, 0)

        r = np.ones((matrix.shape[0], 1))
        t = np.ones(W.shape)
        s = np.ones((matrix.shape[1], 1))

        max_iter = kwargs.get("max_iter", self.max_iter)
        epsilon = kwargs.get("tolerance", self.tolerance)

        for iteration in range(max_iter):
            T = G.T @ t @ Q.T

            pr = (T * P).T @ r
            nr = (
                (np.where(T != 0, 1.0 / T, 1.0) * N).T
                @ self.invd_sparse(r)
                @ np.ones((matrix.shape[0], 1))
            )

            s_new = self.invd_sparse(2 * pr) @ (v + np.sqrt(v**2 + 4 * pr * nr))

            ss = -self.invd_sparse(v) @ nr

            s_new = np.nan_to_num(s_new, nan=1e-10)

            s_new[pr.flatten() == 0] = ss[pr.flatten() == 0]

            ps = (P * T) @ s_new
            ns = (
                (np.where(T != 0, 1.0 / T, 1.0) * N)
                @ self.invd_sparse(s_new)
                @ np.ones((matrix.shape[1], 1))
            )

            r_new = self.invd_sparse(2 * ps) @ (u + np.sqrt(u**2 + 4 * ps * ns))

            rr = -self.invd_sparse(u) @ ns

            r_new = np.nan_to_num(r_new, nan=1e-10)

            r_new[ps.flatten() == 0] = rr[ps.flatten() == 0]

            # update t
            pt = G @ np.diag(r_new.flatten()) @ P @ np.diag(s_new.flatten()) @ Q
            nt = G @ self.invd_sparse(r_new) @ N @ self.invd_sparse(s_new) @ Q

            with np.errstate(divide="ignore", invalid="ignore"):
                t_new = (np.where(pt != 0, 1.0 / pt, 1.0) / 2.0) * (W + np.sqrt(W**2 + 4 * pt * nt))
                W_invd = np.where(W != 0, 1.0 / W, 1.0)
                tr = -nt * W_invd

            t_new[pt == 0] = tr[pt == 0]
            # Handle possible NaNs
            t_new = np.nan_to_num(t_new, nan=1, posinf=1e-10, neginf=1e-10)

            diff = max(np.max(np.abs(r_new - r)), np.max(np.abs(s_new - s)))

            r, s, t = r_new, s_new, t_new

            if diff < epsilon:
                break

        T_final = G.T @ t @ Q.T
        balanced_matrix = self.MRGRAS_UpdateMatrix(P, N, T_final, r, s)

        print(balanced_matrix)
        print(iteration)

        return RASResult(
            balanced_matrix,
            iteration + 1,
            diff < epsilon,
            np.max(np.abs(u - balanced_matrix.sum(axis=1))),
            np.max(np.abs(v - balanced_matrix.sum(axis=0))),
            r,
            s,
        )


def balance_matrix(
    matrix: Union[np.ndarray, sp.spmatrix],
    target_row_sums: np.ndarray,
    target_col_sums: np.ndarray,
    method: str = "RAS",
    bias_matrix: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    bias_method: str = "multiplicative",
    constraints: Optional[List[Set[Tuple[int, int]]]] = None,
    values: Optional[List[float]] = None,
    **kwargs,
) -> RASResult:
    """
    Balances a matrix using the specified method (RAS, GRAS, or MRGRAS).

    Parameters
    ----------
    matrix : Union[np.ndarray, sp.spmatrix]
        The input matrix to balance.
    target_row_sums : np.ndarray
        Target row sums.
    target_col_sums : np.ndarray
        Target column sums.
    method : str, optional
        The balancing method to use ('RAS', 'GRAS', or 'MRGRAS'), by default 'RAS'.
    bias_matrix : Optional[Union[np.ndarray, sp.spmatrix]], optional
        Bias matrix to modify the input matrix before balancing, by default None
    bias_method : str, optional
        Method of applying bias matrix ('multiplicative' or 'additive'),
        by default 'multiplicative'
    constraints : Optional[List[Set[Tuple[int, int]]]], optional
        Constraints for MRGRAS balancing, by default None
    values : Optional[List[float]], optional
        Values for constraints in MRGRAS, by default None
    **kwargs
        Additional parameters passed to the balancer class (e.g., max_iter, tolerance).

    Returns
    -------
    RASResult
        The result of the balancing process.
    """
    if method.upper() == "RAS":
        if bias_matrix is not None:
            warnings.warn("Bias matrix is only supported for GRAS method and will be ignored")
        if constraints is not None or values is not None:
            warnings.warn("Constraints are only supported for MRGRAS method and will be ignored")
        balancer = RASBalancer(**kwargs)
    elif method.upper() == "GRAS":
        if constraints is not None or values is not None:
            warnings.warn("Constraints are only supported for MRGRAS method and will be ignored")
        balancer = GRASBalancer(**kwargs)
    elif method.upper() == "MRGRAS":
        balancer = MRGRASBalancer(**kwargs)
    else:
        raise ValueError(
            f"Unknown balancing method: {method}. Supported methods are 'RAS' and 'MRGRAS'."
        )

    if method.upper() == "MRGRAS":
        return balancer.balance(
            matrix=matrix,
            target_row_sums=target_row_sums,
            target_col_sums=target_col_sums,
            constraints=constraints,
            values=values,
            **kwargs,
        )
    else:
        return balancer.balance(matrix, target_row_sums, target_col_sums, **kwargs)

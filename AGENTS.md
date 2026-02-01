# AGENTS.md

This file contains guidelines for agentic coding agents working in the ras-balancer repository.

## Toolchain Choice: uv and Ruff

We have migrated to a modern Python toolchain using **uv** and **Ruff**.

### Why uv?
- **Speed**: `uv` is significantly faster than Poetry and pip for package resolution and installation.
- **Unified Tool**: It handles Python version management, package management, lockfiles, and virtual environments in a single tool.
- **Standard Compliance**: It uses standard `pyproject.toml` (PEP 621) configuration.

### Why Ruff?
- **Performance**: Ruff is 10-100x faster than existing linters and formatters.
- **Consolidation**: It replaces both `black` (formatting) and `flake8` (linting), as well as `isort` (import sorting), reducing dependency bloat.
- **Compatibility**: It is designed to be a drop-in replacement for these tools with compatible configuration.

## Build / Lint / Test Commands

### Setup
```bash
# Install dependencies
uv sync
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run a single test
uv run pytest tests/test_core.py::TestMatrixBalancers::test_ras_balance_dense_matrix -v

# Run a single test file
uv run pytest tests/test_core.py -v
```

### Formatting & Linting
```bash
# Format code (replaces black)
uv run ruff format .

# Check formatting
uv run ruff format --check .

# Run linter (replaces flake8)
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .
```

### Type Checking
```bash
# Run type checker
uv run mypy .
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

## Code Style Guidelines

### Imports
1. **Order**: Standard library → External packages → Local imports (with relative paths)
2. **Grouping**: Separate each group with a blank line
3. **Local imports**: Use relative imports (e.g., `from .types import BalanceStatus`)
4. **Re-exports**: Use `# noqa: F401` for explicit re-exports in __init__.py

Example:
```python
import os
import logging
from datetime import datetime

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .types import BalanceStatus, RASResult
from .core import RASBalancer
```

### Type System
- **All public functions** must have type hints
- **Arrays**: Use `numpy.typing.NDArray` (e.g., `NDArray[np.float64]`)
- **Matrix parameters**: Use `Union[NDArray, sp.spmatrix]` for both dense and sparse support
- **Optional**: Use `Optional[Type]` for nullable parameters
- **Tuples**: Use `Tuple[Type1, Type2, ...]` for multiple return values

Example:
```python
from typing import Union, Optional
from numpy.typing import NDArray
import scipy.sparse as sp

def balance_matrix(
    matrix: Union[NDArray, sp.spmatrix],
    target_row_sums: NDArray,
    target_col_sums: NDArray,
    bias_matrix: Optional[Union[NDArray, sp.spmatrix]] = None,
) -> RASResult:
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `RASBalancer`, `MatrixGenerator`, `BalanceCheckResult`)
- **Methods/Functions**: `snake_case` (e.g., `balance_matrix`, `generate_balanced_dense`)
- **Private methods**: `_snake_case` prefix (e.g., `_validate_inputs`, `_should_use_sparse`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ITER`, `DEFAULT_TOLERANCE`, `LOG_DIR`)
- **Enum values**: `UPPER_SNAKE_CASE` (e.g., `BalanceStatus.BALANCED`, `ShockType.CELL`)
- **Parameters**: `snake_case` (e.g., `max_iter`, `tolerance`, `target_row_sums`)

### Docstrings
- **Format**: NumPy-style docstrings with Parameters and Returns sections
- **Include**: Brief description, detailed parameter docs, return type docs
- **Method docstrings**: Same format, document self and class parameters if needed

Example:
```python
def balance(
    self,
    matrix: Union[NDArray, sp.spmatrix],
    target_row_sums: NDArray,
    target_col_sums: NDArray,
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

    Returns
    -------
    RASResult
        Results containing balanced matrix and convergence information
    """
```

### Formatting
- **Line length**: 100 characters (configured in pyproject.toml [tool.ruff])
- **Target version**: Python 3.10
- **Formatter**: Ruff (auto-format on commit via pre-commit hooks)

### Error Handling
- **Validation errors**: Raise `ValueError` with descriptive messages for invalid inputs
- **Non-fatal issues**: Use `warnings.warn()` for convergence warnings or non-critical issues
- **Input validation**: Create private `_validate_inputs()` methods to centralize validation logic
- **Division by zero**: Use `np.divide(where=...)` or `np.errstate()` context managers

Example:
```python
def _validate_inputs(self, matrix, target_row_sums, target_col_sums):
    if matrix.shape[0] != len(target_row_sums):
        raise ValueError("Target row sums dimension must match matrix rows")
    if np.any(target_row_sums < 0):
        raise ValueError("Target row sums must be non-negative")
```

### Logging
- **Setup**: Use Python logging module with `logging.getLogger(__name__)`
- **Level**: Configure with `logging.basicConfig(level=logging.INFO)` for module-level
- **Messages**: Use f-strings for formatted log messages
- **Test logging**: Configured in conftest.py with session start/finish hooks

Example:
```python
import logging

logger = logging.getLogger(__name__)

def some_function():
    logger.info(f"Processing matrix of shape {matrix.shape}")
    logger.warning("Matrix did not converge within max iterations")
```

## Testing Conventions

### Test Structure
- **Test classes**: Inherit from no base class, group related tests (e.g., `TestMatrixBalancers`)
- **Fixtures**: Use `@pytest.fixture(autouse=True)` for setup methods that run for all tests
- **File naming**: `tests/test_*.py` matching source files (e.g., `tests/test_core.py`)

### Test Patterns
```python
import pytest
import numpy as np
from ras_balancer import RASBalancer

class TestMatrixBalancers:
    @pytest.fixture(autouse=True)
    def setup_balancers(self):
        """Set up test fixtures."""
        self.balancer = RASBalancer(max_iter=100, tolerance=1e-8)
        self.test_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_ras_balance_dense_matrix(self):
        """Test balancing a dense matrix with RAS."""
        result = self.balancer.balance(self.test_matrix, row_sums, col_sums)
        assert result.converged
        assert np.allclose(result.balanced_matrix.sum(axis=1), row_sums, rtol=1e-6)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        with pytest.raises(ValueError):
            self.balancer.balance(bad_matrix, bad_rows, bad_cols)
```

### Assertions
- **Numpy comparisons**: Use `np.allclose(a, b, rtol=1e-6)` for floating-point arrays
- **Convergence**: Check `result.converged` boolean flag
- **Exceptions**: Use `pytest.raises(ValueError, match="expected_message")` for error cases
- **Matrix shape**: Check `matrix.shape == expected_shape`
- **Sparse check**: Use `sp.issparse(matrix)` for sparse matrix verification

## Project-Specific Patterns

### Dense vs Sparse Matrices
- **Accept both**: Functions should accept `Union[NDArray, sp.spmatrix]`
- **Detection**: Use `sp.issparse(matrix)` to check type
- **Operations**:
  - Dense: `matrix.sum(axis=1)` returns ndarray
  - Sparse: `matrix.sum(axis=1).A1` returns 1D array
  - Conversion: Use `matrix.toarray()` or `sp.csr_matrix(matrix)` as needed
- **Auto-detection**: Implement `_should_use_sparse()` method to determine optimal representation

### Class Structure
- **Base classes**: Create abstract base classes for shared functionality (e.g., `MatrixBalancerBase`)
- **Inheritance**: Subclasses implement specific algorithms (e.g., `RASBalancer`, `GRASBalancer`)
- **Validation**: Move validation logic to base class private methods

### Data Classes
- **Result objects**: Use `@dataclass` for result containers (e.g., `RASResult`, `BalanceCheckResult`)
- **String representation**: Implement `__str__()` methods for readable output
- **Enum types**: Use `Enum` for status codes and type constants (e.g., `BalanceStatus`, `ShockType`)

### Algorithm Implementation
- **RAS algorithm**: Iterative row/column scaling with convergence checking
- **GRAS algorithm**: Handles mixed-sign matrices with positive/negative part decomposition
- **Convergence**: Check both row and column errors against tolerance threshold
- **Chunking**: Process large dense matrices in chunks for memory efficiency (see `chunk_size` parameter)

# Code Duplication Analysis Report: shocker.py

**Generated**: 2026-02-01
**Project**: ras-balancer
**File Analyzed**: `src/ras_balancer/shocker.py`
**Focus**: Code duplication in sparse/dense matrix handling
**Approach**: Test-Driven Development (TDD)
**Severity**: Moderate - Maintainability issue

---

## Executive Summary

**Issue**: Significant code duplication in `shocker.py` around sparse/dense matrix handling patterns, with 8+ repeated `sp.issparse()` checks and duplicated logic blocks.

**Solution**: Extract matrix operations into a new `operations.py` module using Test-Driven Development methodology to eliminate duplication while preserving functionality.

**Status**: **READY FOR IMPLEMENTATION** - TDD approach ensures zero regression while eliminating duplication.

---

## Detailed Analysis

### 1. **Primary Duplication Patterns**

#### **Pattern A: Sparse/Dense Matrix Type Checking**
**Frequency**: 8 occurrences throughout the file

**Problem**: The `sp.issparse(matrix)` check is scattered across multiple methods, making the code harder to maintain and increasing the risk of inconsistent behavior.

**Locations and Impact**:
- **Line 69**: `shock_row_totals()` - Row extraction logic
- **Line 84**: `shock_row_totals()` - Row assignment logic  
- **Line 103**: `shock_column_totals()` - Matrix transpose
- **Line 107**: `shock_column_totals()` - Result transpose
- **Line 121**: `_select_shock_indices()` - Preserve zeros logic
- **Line 124**: `_select_shock_indices()` - Nonzero extraction
- **Line 154**: `shock_proportional()` - Element counting
- **Line 218**: `_calculate_sums()` - Sum calculation

#### **Pattern B: Matrix Transpose Operations**
**Frequency**: 2 nearly identical blocks

**Code Evidence**:
```python
# Pattern 1 (lines 103-104)
shocked = matrix.transpose() if sp.issparse(matrix) else matrix.T

# Pattern 2 (lines 105-109) 
shocked = (
    result.shocked_matrix.transpose()
    if sp.issparse(result.shocked_matrix)
    else result.shocked_matrix.T
)
```

**Problem**: Same transpose logic duplicated with different matrix variables.

#### **Pattern C: Row/Column Extraction and Assignment**
**Frequency**: Multiple similar patterns

**Code Evidence**:
```python
# Line 69 - Extraction pattern
row_values = shocked[idx].toarray().ravel() if sp.issparse(shocked) else shocked[idx]

# Lines 84-87 - Assignment pattern  
if sp.issparse(shocked):
    shocked[idx] = sp.csr_matrix(row_values)
else:
    shocked[idx] = row_values
```

**Problem**: Row extraction and assignment logic repeated across methods.

#### **Pattern D: Element Counting Logic**
**Frequency**: Repeated in multiple shock methods

**Code Evidence** (lines 154-159):
```python
if sp.issparse(matrix):
    n_elements = matrix.nnz
elif self.preserve_zeros:
    n_elements = np.count_nonzero(matrix)
else:
    n_elements = matrix.size
```

**Problem**: Same counting logic duplicated across `shock_proportional()`, `shock_random()`, etc.

#### **Pattern E: Nonzero Element Extraction**
**Frequency**: Similar patterns in index selection

**Code Evidence** (lines 125-127):
```python
if sp.issparse(matrix):
    rows, cols = matrix.nonzero()
else:
    rows, cols = np.where(matrix != 0)
```

**Problem**: Nonzero element extraction logic repeated in multiple places.

---

### 2. **Method-Level Analysis**

#### **Methods with Significant Duplication**

**1. `shock_row_totals()` (lines 57-91)**
- **Issues**: Row extraction and assignment duplication
- **Impact**: 15 lines of sparse/dense handling logic

**2. `shock_column_totals()` (lines 93-113)**
- **Issues**: Matrix transpose duplication
- **Impact**: Complex transpose logic repeated instead of reused

**3. `shock_proportional()` (lines 142-175)**
- **Issues**: Element counting duplication
- **Impact**: Counting logic that should be centralized

**4. `shock_random()` (lines 177-193)**
- **Issues**: Similar structure to `shock_proportional()`
- **Impact**: Potential for further consolidation

#### **Methods with Good Practices (Already Refactored)**

**✅ `_select_shock_indices()` (lines 115-140)**
- **Status**: Well-consolidated index selection logic
- **Pattern**: Good example of proper extraction

**✅ `_calculate_sums()` (lines 216-224)**
- **Status**: Properly consolidated sum calculation
- **Pattern**: Should be model for other extractions

**✅ `_create_shock_result()` (lines 226-248)**
- **Status**: Consistent result creation
- **Pattern**: Good abstraction layer

---

### 3. **Root Cause Analysis**

#### **Why Duplication Exists**:

1. **Historical Development**: Methods were written independently without considering shared patterns
2. **Complex Sparse/Dense Handling**: SciPy sparse matrices require different operations than NumPy dense arrays
3. **Lack of Abstraction Layer**: No unified interface for common matrix operations
4. **Incremental Development**: New methods copied existing patterns instead of extracting common logic

#### **Maintenance Risks**:

1. **Bug Propagation**: Fix in one method may not be applied to others
2. **Inconsistent Behavior**: Slight variations in similar logic can cause different results
3. **Code Readability**: Long methods with repeated sparse/dense checks are hard to follow
4. **Testing Overhead**: Similar logic needs to be tested in multiple places

---

## Detailed Refactoring Plan

## TDD Implementation Plan

### **Phase 1: Test-Driven Development of Matrix Operations Module**

#### **Step 1: Create Complete Test Suite First**
Create `tests/test_operations.py` with comprehensive tests for all helper methods:

```python
# Core operations tests (write BEFORE implementation):
- test_safe_transpose_dense_matrix()
- test_safe_transpose_sparse_matrix()
- test_count_elements_sparse()
- test_count_elements_dense_preserve_zeros()
- test_count_elements_dense_no_preserve()
- test_get_nonzero_indices_sparse()
- test_get_nonzero_indices_dense()
- test_extract_row_sparse()
- test_extract_row_dense()
- test_assign_row_sparse()
- test_assign_row_dense()
- test_integration_operations_together()

# Edge cases:
- test_empty_matrix_operations()
- test_single_element_matrix()
- test_large_sparse_matrix()
- test_mixed_matrix_types()
```

#### **Step 2: Implement operations.py Module Following TDD**
Create `src/ras_balancer/operations.py` with these helper functions:

**Function 1: `safe_transpose(matrix)`**
```python
def safe_transpose(matrix: Union[NDArray, sp.spmatrix]) -> Union[NDArray, sp.spmatrix]:
    """Safely transpose matrix, handling both sparse and dense types."""
```

**Function 2: `count_elements(matrix, preserve_zeros: bool)`**
```python
def count_elements(matrix: Union[NDArray, sp.spmatrix], preserve_zeros: bool) -> int:
    """Count elements according to matrix type and preserve_zeros setting."""
```

**Function 3: `get_nonzero_indices(matrix)`**
```python
def get_nonzero_indices(matrix: Union[NDArray, sp.spmatrix]) -> Tuple[NDArray, NDArray]:
    """Get indices of nonzero elements for both sparse and dense matrices."""
```

**Function 4: `extract_row(matrix, row_index)`**
```python
def extract_row(matrix: Union[NDArray, sp.spmatrix], row_index: int) -> NDArray:
    """Extract row values, handling both sparse and dense matrices."""
```

**Function 5: `assign_row(matrix, row_index, values)`**
```python
def assign_row(matrix: Union[NDArray, sp.spmatrix], row_index: int, values: NDArray) -> None:
    """Assign values to a row, handling both sparse and dense matrices."""
```

#### **Step 3: TDD Implementation Process**
1. **Write one test** → **Run test** (fails) → **Implement minimal code** → **Run test** (passes) → **Refactor**
2. **Repeat** for each helper function
3. **Run complete test suite** to ensure all functions work together

---

### **Phase 2: Refactor shocker.py Using New Operations**

#### **Step 1: Update Imports**
```python
# Add to shocker.py imports
from .operations import (
    safe_transpose,
    count_elements,
    get_nonzero_indices,
    extract_row,
    assign_row,
)
```

#### **Step 2: Refactor Methods One by One**

**Refactor `shock_row_totals()`:**
- Replace line 69: `extract_row(shocked, idx)`
- Replace lines 84-87: `assign_row(shocked, idx, row_values)`

**Refactor `shock_column_totals()`:**
- Replace lines 103-104: `safe_transpose(matrix)`
- Replace lines 105-109: `safe_transpose(result.shocked_matrix)`

**Refactor `shock_proportional()`:**
- Replace element counting (lines 154-159): `count_elements(matrix, self.preserve_zeros)`
- Update `_select_shock_indices()` to use `get_nonzero_indices(matrix)`

**Refactor `shock_random()`:**
- Use same pattern as `shock_proportional()`

---

### **Phase 3: Comprehensive Testing Integration**

#### **Step 1: Run Existing Test Suite**
Ensure no regression in existing functionality:
```bash
uv run pytest tests/test_shocker.py -v
```

#### **Step 2: Add Integration Tests**
Add tests to `tests/test_shocker.py` to verify:
- All shock methods produce identical results before/after refactoring
- Performance hasn't degraded
- Edge cases work correctly with new operations

#### **Step 3: Complete Test Suite Validation**
Run full test suite to verify integration:
```bash
uv run pytest tests/ -v
```

---

## Expected Benefits

### **Immediate Benefits**

1. **Reduced Code Duplication**: Eliminate 8+ `sp.issparse()` checks
2. **Improved Maintainability**: Single place to fix matrix operation bugs
3. **Enhanced Readability**: Cleaner shock methods focused on business logic
4. **Better Testability**: Helper methods can be unit tested independently

### **Long-term Benefits**

1. **Easier Extension**: New shock methods can reuse helper methods
2. **Consistent Behavior**: All methods use identical sparse/dense handling
3. **Reduced Bug Surface**: Fewer places for sparse/dense logic errors
4. **Better Documentation**: Centralized logic is easier to document

### **Metrics for Success**

- **Lines of Code Reduction**: Target ~30-40 lines eliminated
- **Cyclomatic Complexity**: Reduce complexity in shock methods by ~40%
- **Duplication Score**: Eliminate all identified duplication patterns
- **Test Coverage**: Maintain or improve existing test coverage

---

## Risk Assessment

### **Low Risk**
- **Helper Method Creation**: Pure extraction, no logic changes
- **Test Coverage**: Existing tests should pass without modification
- **Performance**: Minimal impact from method call overhead

### **Medium Risk** 
- **Integration Complexity**: Must ensure identical behavior after refactoring
- **Edge Cases**: Sparse/dense boundary conditions need careful testing

### **Mitigation Strategies**
1. **Incremental Development**: Implement one helper method at a time
2. **Comprehensive Testing**: Test each change thoroughly before proceeding
3. **Backup Strategy**: Keep original implementations commented out during development
4. **Code Review**: Peer review of each helper method before integration

---

## Next Steps

1. **Begin Implementation**: Start with Phase 1 TDD approach - write tests first
2. **Create operations.py**: Implement helper functions following test failures
3. **Refactor shocker.py**: Replace duplicated code with new operations
4. **Run Test Suite**: Ensure zero regression with comprehensive testing

---

**Report Generated**: 2026-02-01
**Approach**: Test-Driven Development (TDD)
**Status**: Ready for Implementation
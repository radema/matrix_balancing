# Project Overview

<!-- Include the purpose of the project, how to install, usage examples, and contribution instructions. -->
This repository contains utilities for generating, balancing, and applying shocks to matrices. The core functionality revolves around ensuring matrix row and column sums remain balanced after shocks, and users can introduce shocks at varying levels of precision and randomness.

## Features

* **Matrix Balancing**: Ensures that a matrix's row and column sums match specified totals using the RAS method.
* **Random Matrix Generation**: Provides random balanced matrices, both dense and sparse, with customizable sum and noise levels.
* **Shock Application**: Allows for controlled shocks to be applied to individual matrix elements, rows, or columns, preserving relative proportions or adjusting them based on different distribution strategies.

## Installation

o install the necessary dependencies, you can use pip with pyproject.toml

## Modules
`core.py`
*RASBalancer*: Balances a matrix such that the row and column sums match specified targets, using the RAS (Iterative Proportional Fitting) method.
generator.py
*MatrixGenerator*: Generates random balanced matrices. It provides methods for both dense and sparse matrices, with noise adjustments and balancing.

Methods:

`generate_balanced_dense`: Generates a random balanced dense matrix.
`generate_balanced_sparse`: Generates a random balanced sparse matrix.

`shocker.py`
*MatrixShocker*: Applies shocks to matrices, modifying individual cells, rows, or columns. It allows for both relative and absolute shocks and can preserve zeros if specified.

Methods:

`shock_cell`: Shocks a specific matrix cell.
`shock_row_totals`: Shocks row totals while maintaining relative proportions.
`shock_column_totals`: Shocks column totals while maintaining relative proportions.
`shock_proportional`: Applies proportional shocks to a fraction of non-zero elements.
`shock_random`: Randomly applies shocks to matrix elements.

`types.py`
Defines various types and dataclasses:

*BalanceStatus*: Enumeration for matrix balance status (e.g., BALANCED, UNBALANCED_ROWS).
*ShockType*: Enumeration for different types of matrix shocks (e.g., CELL, ROW_TOTAL).
*BalanceCheckResult*: Data class to store the results of a matrix balance check.
*RASResult*: Data class to store the results of the RAS algorithm.
*ShockResult*: Data class to store the results of matrix shock application.

## Usage

Below is a basic example of how to use the classes in this repository.

```python
from core import RASBalancer
from generator import MatrixGenerator
from shocker import MatrixShocker

# Generate a random balanced matrix
matrix, row_sums, col_sums = MatrixGenerator.generate_balanced_dense(5, 5)

# Apply a shock to a specific cell
shocker = MatrixShocker()
result = shocker.shock_cell(matrix, 0, 0, magnitude=0.1)

# Print shocked matrix
print(result.shocked_matrix)
```

## License
This project is licensed under the MIT License - see the [LICENSE] file for details.
"""
Test suite for the RAS Balancer package.

This package contains unit tests for:
- Core RAS balancing functionality
- Matrix generation utilities
- Matrix shocking capabilities
"""

import os
import sys

# Add the src directory to the Python path for testing
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

"""
KL controllers for SAT-Seq recipe.

This subpackage provides quantile-based KL control with:
  - Per-agent adaptive β coefficients
  - Quantile-based constraint checking
  - Backtracking when constraints are violated
"""

__all__ = ["QuantileKLController", "KLStats"]


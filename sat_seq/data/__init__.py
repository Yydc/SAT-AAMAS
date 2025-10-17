"""
Data processing utilities for SAT-Seq recipe.

This subpackage provides truncated importance sampling (IS) reweighting for:
  - Sequential policy updates with inter-policy distribution shifts
  - Variance reduction via truncation
  - Numerical stability for long sequences
"""

__all__ = ["TruncatedISReweighter"]


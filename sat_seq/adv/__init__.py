"""
Advantage estimators for SAT-Seq recipe.

This subpackage provides sequence-aware advantage estimation with:
  - Group-based normalization
  - Clipping to prevent extreme outliers
  - Bias tracking for truncated IS reweighting
"""

__all__ = ["SATSeqAdvEstimator"]


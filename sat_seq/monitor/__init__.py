"""
Monitoring and certificate tracking for SAT-Seq recipe.

This subpackage provides PAC-style certificate tracking with:
  - Information gain accumulation across sequential updates
  - Penalty terms for distribution shift, estimator bias, and finite samples
  - Lower-bound guarantees for policy improvement
"""

__all__ = ["CertificateMonitor", "InfoGeom"]


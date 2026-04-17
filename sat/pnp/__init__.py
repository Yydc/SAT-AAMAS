"""Plug-and-play utilities for SAT.

Provides the Stage-0 KL projection that initialises a swapped-in pretrained
agent inside the per-state trust region of the current team member, keeping
the monotonic-improvement certificate of Theorems 1.1-1.2 valid.
"""

from sat.pnp.stage0_alignment import Stage0Aligner

__all__ = ["Stage0Aligner"]

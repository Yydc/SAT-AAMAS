"""
Policy loss functions for SAT-Seq recipe.

This subpackage provides sequence-level policy gradient objectives with:
  - Sequence-level ratio clipping (PPO-style)
  - Active-token masking for multi-agent updates
  - KL penalty integration
"""

__all__ = ["SeqRatioPolicyLoss"]


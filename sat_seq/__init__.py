"""
SAT-Seq Recipe for veRL: Sequence-Aware Block-Coordinate Tuning.

This package extends veRL with a multi-agent, stage-wise training recipe that:
  - Maintains separate expert agents (e.g., three Qwen3-4B models)
  - Optimizes agents sequentially within each stage via block-coordinate descent
  - Employs sequence-level advantages, quantile-based KL control, and truncated IS
  - Provides PAC-style certificate tracking for lower-bound guarantees

Main components:
  * StageCoordinator: orchestrates a single SAT stage (on-policy rollout + sequential agent updates)
  * AgentScheduler: decides the order in which agents are updated within a stage
  * CertificateMonitor: accumulates per-agent records and estimates the final lower bound

Integration with veRL:
  - Use the configuration in config/sat_seq.yaml as a starting point
  - Register custom components (advantage estimators, KL controllers, policy losses) as noted below
  - Inject StageCoordinator into a top-level training loop (see README.md for details)
"""

__all__ = [
    "StageCoordinator",
    "AgentScheduler",
    "CertificateMonitor",
]

# Type annotations only; do NOT import real modules here to avoid circular dependencies.
# Users should import from submodules explicitly:
#   from recipe.sat_seq.stage_coordinator import StageCoordinator
#   from recipe.sat_seq.agent_scheduler import AgentScheduler
#   from recipe.sat_seq.monitor.certificate import CertificateMonitor

# ============================================================================
# REGISTRY TODO: Manual registration required in veRL's internal registries
# ============================================================================
#
# 1) Advantage Estimator
#    Key: "sat_seq"
#    Class path: "recipe.sat_seq.adv.seqaware.SATSeqAdvEstimator"
#    Where to register:
#      - Typically in veRL's trainer or config system (e.g., a factory dict for adv_estimator types)
#      - Example (pseudo-code):
#          ADV_ESTIMATOR_REGISTRY["sat_seq"] = "recipe.sat_seq.adv.seqaware.SATSeqAdvEstimator"
#
# 2) KL Controller
#    Type: "per_state_quantile"
#    Class path: "recipe.sat_seq.kl.quantile_kl_ctrl.QuantileKLController"
#    Where to register:
#      - In veRL's kl_ctrl factory or config dispatcher
#      - Example (pseudo-code):
#          KL_CTRL_REGISTRY["per_state_quantile"] = "recipe.sat_seq.kl.quantile_kl_ctrl.QuantileKLController"
#
# 3) Policy Loss Builder
#    Class path: "recipe.sat_seq.loss.seq_ratio_loss.SeqRatioPolicyLoss"
#    Where to register:
#      - In veRL's policy loss builder registry or trainer hooks
#      - Example (pseudo-code):
#          POLICY_LOSS_REGISTRY["seq_ratio"] = "recipe.sat_seq.loss.seq_ratio_loss.SeqRatioPolicyLoss"
#
# Please update the corresponding registration files or configuration loaders
# in the veRL codebase to ensure these components are discoverable at runtime.
# ============================================================================

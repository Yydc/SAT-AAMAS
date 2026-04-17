"""Sequential Agent Tuning (SAT) - official implementation.

SAT is a coordinator-free training paradigm for a factorized team of LLMs.
At each stage it rolls out the current team, picks an update order, and
applies sequential trust-region updates under a per-agent per-state KL
radius. The package exposes the core modules used by the top-level scripts:

    * ``sat.stage_coordinator.StageCoordinator``  -- Algorithm 1 outer loop
    * ``sat.agent_scheduler.AgentScheduler``      -- ORDERAGENTS strategies
    * ``sat.real_controller.RealMultiAgentController``
                                                  -- HF-backed multi-agent rollout/update
    * ``sat.monitor.certificate.CertificateMonitor``
                                                  -- PAC-style stage bound
    * ``sat.pnp.stage0_alignment.Stage0Aligner``  -- plug-and-play KL projection
"""

__version__ = "0.1.0"

__all__ = [
    "StageCoordinator",
    "AgentScheduler",
    "CertificateMonitor",
    "RealMultiAgentController",
    "Stage0Aligner",
]

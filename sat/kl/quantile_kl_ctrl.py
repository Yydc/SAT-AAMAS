"""Quantile KL Controller - KL divergence control based on quantiles."""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class KLStats:
    """KL divergence statistics."""
    q_val: float      # Quantile value
    samples: int      # Number of samples
    mean: float       # Mean value


@dataclass
class QuantileKLController:
    """Quantile KL controller that dynamically adjusts the KL coefficient β."""
    
    config: Dict
    _kl_coef_cache: Dict[int, float] = field(default_factory=dict)

    def current_kl_coef(self, agent_id: int) -> float:
        """Gets the current KL coefficient β."""
        if agent_id not in self._kl_coef_cache:
            default_kl_coef = self.config["algorithm"]["kl_ctrl"].get("kl_coef", 0.001)
            self._kl_coef_cache[agent_id] = default_kl_coef
        return self._kl_coef_cache[agent_id]

    def target_delta(self, agent_id: int) -> float:
        """Gets the target KL delta."""
        return self.config["algorithm"]["kl_ctrl"].get("target_kl", 0.05)

    def measure_and_control(self, controller, agent_id: int) -> KLStats:
        """Measures KL divergence and returns statistics."""
        kl_vector = controller.get_per_state_kl(agent_id)
        
        if not isinstance(kl_vector, np.ndarray):
            # Fallback for non-numpy
            return KLStats(q_val=0.01, samples=100, mean=0.01)
        
        quantile_val = self.config["algorithm"]["kl_ctrl"].get("quantile", 0.99)
        q_val = np.quantile(kl_vector, quantile_val)
        
        return KLStats(
            q_val=float(q_val),
            samples=len(kl_vector),
            mean=float(np.mean(kl_vector))
        )

    def increase_pressure(self, agent_id: int) -> None:
        """Increases the KL penalty coefficient (called during backtracking)."""
        current_beta = self._kl_coef_cache.get(
            agent_id, 
            self.config["algorithm"]["kl_ctrl"].get("kl_coef", 0.001)
        )
        new_beta = current_beta * 1.5
        self._kl_coef_cache[agent_id] = new_beta

    def effective_delta(self, agent_id: int, stats: KLStats) -> float:
        """Returns the effective KL delta (can incorporate DKW relaxation)."""
        return stats.q_val

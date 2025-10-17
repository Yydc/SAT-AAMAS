"""Truncated IS Reweighter - Truncated Importance Sampling Reweighting."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class TruncatedISReweighter:
    """Truncated Importance Sampling Reweighter, calculates c_t = min(1, ρ_t)."""
    
    config: Dict

    def __post_init__(self):
        self.epsilon_is = self.config["sat_seq"].get("epsilon_is", 1e-5)
        self.max_ratio_clip = self.config["sat_seq"].get("max_is_ratio_clip", 10.0)

    def apply(self, stage_batch: Dict, inter_policy_state: Dict) -> Dict:
        """
        Applies truncated IS reweighting.
        
        Args:
            stage_batch: The original batch data.
            inter_policy_state: Intermediate state information.
            
        Returns:
            A batch with the 'weights_is' field added (shallow copy).
        """
        reweighted_batch = stage_batch.copy()
        reweighted_batch["meta"] = stage_batch.get("meta", {}).copy()
        
        # Calculate ρ_t = exp(logp_hat - logp_cur)
        logp_cur = stage_batch.get("logp_cur")
        logp_hat = stage_batch.get("logp_hat", logp_cur)  # Use logp_cur if logp_hat is not available
        
        if isinstance(logp_cur, np.ndarray) and isinstance(logp_hat, np.ndarray):
            log_ratio = logp_hat - logp_cur
            log_ratio_clipped = np.clip(log_ratio, -self.max_ratio_clip, self.max_ratio_clip)
            rho_t = np.exp(log_ratio_clipped)
            is_weights = np.minimum(1.0, rho_t)
            
            # Statistics
            truncation_rate = (rho_t > 1.0).mean()
            max_ratio = rho_t.max()
            mean_ratio = rho_t.mean()
        else:
            # Fallback: all-one weights
            shape = logp_cur.shape if hasattr(logp_cur, 'shape') else (100, 100)
            is_weights = np.ones(shape)
            truncation_rate, max_ratio, mean_ratio = 0.0, 1.0, 1.0
        
        reweighted_batch["weights_is"] = is_weights
        reweighted_batch["meta"]["is_truncation_rate"] = float(truncation_rate)
        reweighted_batch["meta"]["is_max_ratio"] = float(max_ratio)
        reweighted_batch["meta"]["is_mean_ratio"] = float(mean_ratio)
        
        return reweighted_batch

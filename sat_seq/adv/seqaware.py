"""Sequence-Aware Advantage Estimator - Computes sequence-level advantages."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SATSeqAdvEstimator:
    """Sequence-aware advantage estimator, supporting GAE and GRPO modes.
    - GAE: Based on token-level (r - V) approximation, aggregated to the sequence level.
    - GRPO: Directly uses sequence returns with an in-group baseline (mean/max), with optional normalization.
    """
    
    config: Dict

    def __post_init__(self):
        self.A_clip = self.config["sat_seq"].get("A_clip", 5.0)
        self.group_size = self.config["sat_seq"].get("group_size", 4)
        self.lam = self.config["algorithm"].get("lam", 0.95)
        self.gamma = self.config["algorithm"].get("gamma", 0.99)
        # GRPO-related config
        self.adv_mode = self.config.get("sat_seq", {}).get("adv_mode", "grpo")  # grpo | gae
        self.group_baseline = self.config.get("sat_seq", {}).get("group_baseline", "mean")  # mean|max
        self.group_norm = bool(self.config.get("sat_seq", {}).get("group_norm", True))

    def compute(self, stage_batch_i: Dict, inter_policy_state: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computes sequence-level advantages and a bias estimate.
        
        Args:
            stage_batch_i: A batch containing rewards, values, weights_is, etc.
            inter_policy_state: Intermediate state information.
            
        Returns:
            (seq_adv, returns, zeta_i): Sequence advantage array, returns array, and bias proxy.
        """
        # Data
        rewards = stage_batch_i.get("rewards")
        values = stage_batch_i.get("values")
        num_episodes = stage_batch_i.get("meta", {}).get("num_episodes", 10)
        group_index = stage_batch_i.get("meta", {}).get("group_index")

        # ========== GRPO: Group-wise baseline based on sequence returns ==========
        if self.adv_mode == "grpo":
            # 1) Calculate per-sequence return (sum over tokens; if reward is only at the last token, this is the last token's value)
            if isinstance(rewards, np.ndarray):
                if rewards.ndim == 2:
                    seq_return = rewards.sum(axis=1)
                else:
                    seq_return = rewards
            else:
                seq_return = np.zeros(num_episodes, dtype=np.float32)

            # 2) In-group baseline
            if group_index is not None:
                adv = np.copy(seq_return)
                for gid in np.unique(group_index):
                    mask = (group_index == gid)
                    group_vals = seq_return[mask]
                    if len(group_vals) == 0:
                        continue
                    if self.group_baseline == "max":
                        baseline = np.max(group_vals)
                    else:
                        baseline = np.mean(group_vals)
                    adv[mask] = group_vals - baseline

                # 3) In-group normalization (optional)
                if self.group_norm:
                    for gid in np.unique(group_index):
                        mask = (group_index == gid)
                        group_adv = adv[mask]
                        if len(group_adv) > 1:
                            mean = group_adv.mean()
                            std = group_adv.std() + 1e-8
                            adv[mask] = (group_adv - mean) / std
            else:
                # No group info: use the global mean as baseline
                baseline = float(np.mean(seq_return)) if len(seq_return) > 0 else 0.0
                adv = seq_return - baseline
                if self.group_norm and len(seq_return) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # 4) Clipping
            seq_adv = np.clip(adv, -self.A_clip, self.A_clip)
            zeta_i = 0.0
            
            # In GRPO mode, returns are equal to the sequence returns
            returns = seq_return
            
            return seq_adv, returns, zeta_i

        # ========== GAE: Generalized Advantage Estimation ==========
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        # GAE calculation
        for t in reversed(range(rewards.shape[1])):
            if t == rewards.shape[1] - 1:
                next_value = 0  # Assume end of sequence
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.gamma * next_value - values[:, t]
            last_advantage = delta + self.gamma * self.lam * last_advantage
            advantages[:, t] = last_advantage
            
        returns = advantages + values

        # Aggregate to sequence level (sum over tokens)
        if isinstance(advantages, np.ndarray):
            seq_adv = advantages.sum(axis=1) if len(advantages.shape) > 1 else advantages
        else:
            seq_adv = np.zeros(num_episodes)
        
        # Group normalization (normalize responses for the same prompt)
        if group_index is not None:
            seq_adv = self._group_normalize(seq_adv, group_index)
        
        # Clip to ±A_clip
        seq_adv = np.clip(seq_adv, -self.A_clip, self.A_clip)
        
        # Bias proxy (placeholder)
        # TODO: Implement precise calculation of zeta_i based on the SAT-Seq paper
        zeta_i = float(np.mean(np.abs(advantages.sum(axis=1) - seq_adv))) if group_index is not None else 0.0
        
        return seq_adv, returns, zeta_i
    
    def _group_normalize(self, seq_adv: np.ndarray, group_index: np.ndarray) -> np.ndarray:
        """In-group normalization (standardize responses within the same prompt group)."""
        normalized = np.copy(seq_adv)
        for group_id in np.unique(group_index):
            mask = group_index == group_id
            group_values = seq_adv[mask]
            if len(group_values) > 1:
                mean = group_values.mean()
                std = group_values.std() + 1e-8
                normalized[mask] = (group_values - mean) / std
        return normalized

"""Sequence Ratio Policy Loss - A PPO-style, sequence-level policy loss."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SeqRatioPolicyLoss:
    """
    Sequence-level ratio loss that implements a PPO-like clipped objective
    and a value function loss.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.epsilon = self.config["sat_seq"].get("epsilon", 0.2)
        self.loss_agg_mode = self.config["actor_rollout_ref"]["actor"].get("loss_agg_mode", "token-mean")
        self.vf_coef = self.config.get("training", {}).get("vf_coef", 0.1)
        self.clip_vf = self.config.get("training", {}).get("clip_vf", True)

    def forward(self, stage_batch_i: Dict, inter_policy_state: Dict, 
                agent_id: int, seq_adv: np.ndarray, kl_coef: float, returns: np.ndarray) -> Dict:
        """
        Calculates the PPO loss (policy loss + value loss).
        
        Args:
            stage_batch_i: ...
            inter_policy_state: ...
            agent_id: ...
            seq_adv: ...
            kl_coef: ...
            returns: np.ndarray, target values (returns), shape [N, T]
            
        Returns:
            A dictionary containing "loss" (policy loss) and "value_loss".
        """
        import torch
        import torch.nn.functional as F

        # Get required fields
        logp_old_np = stage_batch_i.get("logp_cur")  # [N, T]
        values_old_np = stage_batch_i.get("values") # [N, T]
        prompt_ids_list = stage_batch_i.get("prompt_ids", [])
        response_ids_list = stage_batch_i.get("response_ids", [])
        resp_lens = stage_batch_i.get("meta", {}).get("response_len")
        controller = inter_policy_state.get("controller")

        if controller is None:
            raise RuntimeError("Controller not injected into inter_policy_state, cannot compute new policy logp")

        # Use the controller to compute the new policy's logp and value
        logp_new, value_new = controller.compute_logprobs_and_values_for_batch(
            agent_id, prompt_ids_list, response_ids_list, resp_lens
        )
        
        if not isinstance(logp_old_np, np.ndarray):
            raise RuntimeError("stage_batch_i.logp_cur is missing or has the wrong type")
        logp_old = torch.tensor(logp_old_np, dtype=torch.float32, device=logp_new.device)
        T = min(logp_new.shape[1], logp_old.shape[1])
        logp_new = logp_new[:, :T]
        logp_old = logp_old[:, :T]

        # Calculate the cumulative log-ratio per sequence: u_i = sum_t (logp_new - logp_old)
        u_seq = torch.sum(logp_new - logp_old, dim=1)  # [N]
        r_seq = torch.exp(u_seq)  # [N]

        # Convert advantage to a torch tensor
        if isinstance(seq_adv, np.ndarray):
            A_seq = torch.tensor(seq_adv, dtype=torch.float32, device=logp_new.device)
        else:
            A_seq = torch.zeros(logp_new.shape[0], dtype=torch.float32, device=logp_new.device)

        # PPO-style clipping
        eps = float(self.epsilon)
        r_clipped = torch.clamp(r_seq, 1.0 - eps, 1.0 + eps)
        obj1 = r_seq * A_seq
        obj2 = r_clipped * A_seq
        ppo_obj = torch.min(obj1, obj2)

        # Sequence-level aggregation (negated to form a loss)
        loss = -torch.mean(ppo_obj)

        # ========== Calculate Value Loss ==========
        values_old = torch.tensor(values_old_np[:, :T], dtype=torch.float32, device=value_new.device)
        returns = torch.tensor(returns[:, :T], dtype=torch.float32, device=value_new.device)
        
        if self.clip_vf:
            value_new_clipped = torch.clamp(
                value_new,
                values_old - self.epsilon,
                values_old + self.epsilon,
            )
            vf_loss1 = F.mse_loss(value_new, returns, reduction='none')
            vf_loss2 = F.mse_loss(value_new_clipped, returns, reduction='none')
            value_loss = 0.5 * torch.mean(torch.maximum(vf_loss1, vf_loss2))
        else:
            value_loss = 0.5 * F.mse_loss(value_new, returns)

        return {
            "loss": loss,
            "value_loss": value_loss,
            "aux": {
                "ratio_mean": r_seq.mean().detach().item(),
                "adv_mean": A_seq.mean().detach().item(),
                "epsilon": eps,
                "kl_coef": float(kl_coef),
                "value_mean": value_new.mean().detach().item(),
                "return_mean": returns.mean().detach().item(),
            }
        }

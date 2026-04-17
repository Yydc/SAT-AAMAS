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
        sat_cfg = config.get("sat") or config.get("sat_seq") or {}
        self.epsilon = sat_cfg.get("epsilon", 0.2)
        actor_cfg = config.get("actor_rollout_ref", {}).get("actor", {})
        self.loss_agg_mode = actor_cfg.get("loss_agg_mode", "token-mean")
        self.vf_coef = config.get("training", {}).get("vf_coef", 0.1)
        self.clip_vf = config.get("training", {}).get("clip_vf", True)

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
        rollout_agent_ids = stage_batch_i.get("meta", {}).get("agent_id")
        controller = inter_policy_state.get("controller")

        if controller is None:
            raise RuntimeError("Controller not injected into inter_policy_state, cannot compute new policy logp")

        if isinstance(rollout_agent_ids, np.ndarray):
            keep = np.where(rollout_agent_ids == agent_id)[0]
            if len(keep) == 0:
                return {
                    "loss": torch.tensor(0.0),
                    "value_loss": torch.tensor(0.0),
                    "skip": True,
                    "aux": {"reason": "no samples for active agent"},
                }
            logp_old_np = logp_old_np[keep]
            values_old_np = values_old_np[keep]
            prompt_ids_list = [prompt_ids_list[i] for i in keep]
            response_ids_list = [response_ids_list[i] for i in keep]
            resp_lens = resp_lens[keep] if resp_lens is not None else None
            if isinstance(seq_adv, np.ndarray):
                seq_adv = seq_adv[keep]
            returns = np.asarray(returns)
            returns = returns[keep]

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
        value_new = value_new[:, :T]

        if resp_lens is None:
            lengths = torch.full((logp_new.shape[0],), T, dtype=torch.long, device=logp_new.device)
        else:
            lengths = torch.tensor(resp_lens, dtype=torch.long, device=logp_new.device).clamp(min=1, max=T)
        token_mask = torch.arange(T, device=logp_new.device).unsqueeze(0) < lengths.unsqueeze(1)
        token_mask_f = token_mask.float()

        # Calculate the cumulative log-ratio per sequence: u_i = sum_t (logp_new - logp_old)
        log_ratio_tok = torch.clamp(logp_new - logp_old, -10.0, 10.0)
        u_seq = torch.sum(log_ratio_tok * token_mask_f, dim=1)  # [N]
        r_seq = torch.exp(torch.clamp(u_seq, -20.0, 20.0))  # [N]

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

        # Sequence-level aggregation (negated to form a loss), plus the
        # per-agent sampled KL penalty from the manuscript surrogate.
        kl_tok = (torch.exp(log_ratio_tok) - 1.0) - log_ratio_tok
        kl_loss = (kl_tok * token_mask_f).sum() / token_mask_f.sum().clamp_min(1.0)
        loss = -torch.mean(ppo_obj) + float(kl_coef) * kl_loss

        # ========== Calculate Value Loss ==========
        values_old = torch.tensor(values_old_np[:, :T], dtype=torch.float32, device=value_new.device)
        # In GRPO mode the advantage estimator returns one scalar per sequence;
        # broadcast it across timesteps so the value head fits the same target
        # at every active token (paper Section 4 "Group-relative normalization").
        returns_arr = np.asarray(returns)
        if returns_arr.ndim == 1:
            returns_arr = np.broadcast_to(returns_arr[:, None], (returns_arr.shape[0], T))
        else:
            returns_arr = returns_arr[:, :T]
        returns = torch.tensor(returns_arr, dtype=torch.float32, device=value_new.device)
        
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
                "kl_loss": kl_loss.detach().item(),
                "value_mean": value_new.mean().detach().item(),
                "return_mean": returns.mean().detach().item(),
            }
        }

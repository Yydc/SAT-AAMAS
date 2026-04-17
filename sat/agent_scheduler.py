"""Agent Scheduler - Determines the update order of agents."""

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AgentScheduler:
    """
    Schedules the update order of agents.
    
    Supports three modes:
    - static: In order of ID [0,1,2,...]
    - random: Shuffled randomly
    - greedy_info_gain: Descending order of predicted information gain
    """
    mode: str = "static"
    seed: int = None

    def __post_init__(self):
        if self.mode not in {"static", "random", "greedy_info_gain"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.seed is not None:
            random.seed(self.seed)

    def order_agents(self, stage_batch: Dict, policy_stats: Dict) -> List[int]:
        """
        Determines the agent update order.
        
        Args:
            stage_batch: The data batch for the current stage.
            policy_stats: A dictionary containing num_agents and optionally pred_gain.
            
        Returns:
            A list of agent IDs.
        """
        num_agents = policy_stats.get("num_agents", 0)
        if num_agents <= 0:
            return []

        if self.mode == "static":
            return list(range(num_agents))
        
        elif self.mode == "random":
            order = list(range(num_agents))
            random.shuffle(order)
            return order
        
        elif self.mode == "greedy_info_gain":
            pred_gain = policy_stats.get("pred_gain")
            if pred_gain is None or len(pred_gain) != num_agents:
                # Fallback to static mode
                return list(range(num_agents))
            # Sort by pred_gain in descending order
            order_with_gain = sorted(enumerate(pred_gain), key=lambda x: x[1], reverse=True)
            return [agent_id for agent_id, _ in order_with_gain]
        
        return list(range(num_agents))

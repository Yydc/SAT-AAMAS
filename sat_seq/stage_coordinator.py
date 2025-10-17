"""SAT-Seq Stage Coordinator - Orchestrates the end-to-end process of a single training stage."""

from dataclasses import dataclass
from typing import Dict, List

from recipe.sat_seq.adv.seqaware import SATSeqAdvEstimator
from recipe.sat_seq.data.reweighting import TruncatedISReweighter
from recipe.sat_seq.kl.quantile_kl_ctrl import QuantileKLController
from recipe.sat_seq.loss.seq_ratio_loss import SeqRatioPolicyLoss


@dataclass
class InfoGeom:
    """Information geometry coefficients (Fisher matrix estimates, currently placeholders)."""
    kappa_reg: float = 0.0
    a_reg: float = 0.0


class StageCoordinator:
    """
    Coordinates the execution of a single SAT-Seq training stage.
    
    Process: Collect data -> Determine agent order -> Sequentially update each agent -> Calculate certificate
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.max_backtracks = cfg.get("sat_seq", {}).get("max_backtracks", 5)
        
        # Initialize components
        self.reweighter = TruncatedISReweighter(cfg)
        self.adv_estimator = SATSeqAdvEstimator(cfg)
        self.kl_controller = QuantileKLController(cfg)
        self.policy_loss = SeqRatioPolicyLoss(cfg)

    def run_one_stage(self, controller, scheduler, cert_monitor) -> Dict:
        """Executes a complete SAT-Seq stage."""
        
        # 1. Collect on-policy data
        stage_batch = self._collect_on_policy(controller)
        # Save the latest batch for KL/loss calculation
        if hasattr(controller, "set_last_stage_batch"):
            controller.set_last_stage_batch(stage_batch)

        # 2. Determine agent update order
        policy_stats = {"num_agents": controller.num_agents()}
        order = scheduler.order_agents(stage_batch, policy_stats)

        if len(order) != controller.num_agents():
            raise ValueError(f"Scheduler returned {len(order)} agents, but expected {controller.num_agents()}")

        # 3. Sequentially update each agent
        per_agent_records = []
        for step_i, agent_id in enumerate(order, start=1):
            print(f"[Stage] Updating agent {agent_id} ({step_i}/{len(order)})")

            # Activate the current agent
            controller.activate_agent(agent_id)

            # Reweighting
            inter_state = {"order": order, "step": step_i, "controller": controller}
            stage_batch_i = self.reweighter.apply(stage_batch, inter_state)

            # Calculate advantage and returns
            seq_adv, returns, zeta_i = self.adv_estimator.compute(stage_batch_i, inter_state)

            # Backtracking optimization loop
            backtrack_count = 0
            for bt_iter in range(self.max_backtracks + 1):
                kl_coef = self.kl_controller.current_kl_coef(agent_id)
                loss_out = self.policy_loss.forward(
                    stage_batch_i, inter_state, agent_id, seq_adv, kl_coef, returns
                )
                controller.optimize_step(loss_out)
                
                kl_stats = self.kl_controller.measure_and_control(controller, agent_id)
                target_delta = self.kl_controller.target_delta(agent_id)
                
                if kl_stats.q_val <= target_delta:
                    print(f"  KL constraint met: {kl_stats.q_val:.6f} <= {target_delta:.6f}")
                    break

                if bt_iter < self.max_backtracks:
                    print(f"  KL exceeded: {kl_stats.q_val:.6f} > {target_delta:.6f}, backtracking...")
                    controller.backtrack_last_step()
                    self.kl_controller.increase_pressure(agent_id)
                    backtrack_count += 1
                else:
                    print("  Max backtracks reached")

            # Record agent statistics
            info_geom = InfoGeom()  # placeholder
            per_agent_records.append({
                "agent_id": agent_id,
                "delta_i": self.kl_controller.effective_delta(agent_id, kl_stats),
                "N_i": stage_batch_i.get("meta", {}).get("num_episodes", 1),
                "zeta_i": zeta_i,
                "kappa": info_geom.kappa_reg,
                "a": info_geom.a_reg,
                "backtrack_count": backtrack_count,
            })

        # 4. Calculate stage certificate
        stage_result = cert_monitor.finish_stage(per_agent_records)
        print(f"[Stage] Completed, lower bound: {stage_result['lower_bound']:.6f}")
        
        return stage_result

    def _collect_on_policy(self, controller) -> Dict:
        """Collects on-policy rollout data."""
        stage_batch = controller.generate_sequences(
            batch_size=self.cfg.get("data", {}).get("train_batch_size", 512),
            max_seq_len=self.cfg.get("data", {}).get("max_response_length", 128),
        )
        
        required_keys = ["prompts", "responses", "rewards", "logp_cur", "values", "meta"]
        for key in required_keys:
            if key not in stage_batch:
                raise KeyError(f"stage_batch is missing required key: {key}")
        
        print(f"  Collected {stage_batch['meta']['num_episodes']} episodes")
        return stage_batch

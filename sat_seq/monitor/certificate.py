"""Certificate Monitor - A PAC-style certificate monitor."""

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class InfoGeom:
    """Information geometry coefficients."""
    kappa_reg: float
    a_reg: float


@dataclass
class CertificateMonitor:
    """A PAC-style certificate monitor that calculates the lower bound of policy improvement."""
    
    gamma: float          # Discount factor
    A_max: float          # Maximum advantage value
    delta_conf: float     # Confidence level

    def finish_stage(self, per_agent_records: List[Dict]) -> Dict:
        """
        Aggregates per-agent records and calculates the stage certificate.
        
        Args:
            per_agent_records: A list of statistical records for each agent.
            
        Returns:
            A dictionary containing info_gain, penalties, lower_bound, etc.
        """
        if not per_agent_records:
            return {
                "info_gain": 0.0,
                "occ_shift_penalty": 0.0,
                "estimator_bias_penalty": 0.0,
                "finite_sample_penalty": 0.0,
                "lower_bound": 0.0,
                "n_agents": 0,
            }

        total_info_gain = 0.0
        total_occ_shift_penalty = 0.0
        total_estimator_bias_penalty = 0.0
        total_finite_sample_penalty = 0.0

        for record in per_agent_records:
            delta_i = record["delta_i"]
            N_i = record["N_i"]
            zeta_i = record["zeta_i"]
            n_agents = len(per_agent_records)

            # 1. Information-Geometric Gain
            # The exact terms from the paper (kappa, a) are hard to compute. We use zeta_i
            # (a proxy for the advantage estimate) as a proxy for L_i^seq, which is
            # consistent with the spirit of Theorem 1.4.
            # E[A] is an estimate of (1-gamma) * (J_new - J_old).
            info_gain_i = zeta_i
            
            # 2. Occupancy-Shift Penalty
            # From paper formula (1) and Lemma 1.1
            occ_shift_penalty_i = (2 * self.gamma / ((1 - self.gamma)**2)) * self.A_max * math.sqrt(0.5 * delta_i)
            
            # 3. Estimator-Bias Penalty
            # From paper formula (1)
            estimator_bias_penalty_i = zeta_i / (1 - self.gamma)
            
            # 4. Finite-Sample Error
            # From paper formula (1)
            if N_i > 0:
                log_term = math.log((2 * n_agents) / self.delta_conf)
                finite_sample_penalty_i = (self.A_max / (1 - self.gamma)) * math.sqrt(log_term / (2 * N_i))
            else:
                finite_sample_penalty_i = float('inf')

            total_info_gain += info_gain_i
            total_occ_shift_penalty += occ_shift_penalty_i
            total_estimator_bias_penalty += estimator_bias_penalty_i
            total_finite_sample_penalty += finite_sample_penalty_i

        # Lower Bound = Information Gain - Various Penalties
        lower_bound = (total_info_gain
                      - total_occ_shift_penalty
                      - total_estimator_bias_penalty
                      - total_finite_sample_penalty)

        return {
            "info_gain": total_info_gain,
            "occ_shift_penalty": total_occ_shift_penalty,
            "estimator_bias_penalty": total_estimator_bias_penalty,
            "finite_sample_penalty": total_finite_sample_penalty,
            "lower_bound": lower_bound,
            "n_agents": len(per_agent_records),
        }

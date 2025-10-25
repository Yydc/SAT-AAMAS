# SAT-Seq: Sequence-Aware Block-Coordinate Tuning

## Overview

This repository contains the official implementation of SAT-Seq, a multi-agent reinforcement learning (MARL) framework that employs block-coordinate descent (BCD) for sequential agent optimization. The algorithm provides Probably Approximately Correct (PAC) style theoretical guarantees on performance.

**Note:** This recipe is designed for the verl library and should be placed within the `verl/recipes/` directory.

## Methodology

SAT-Seq addresses the challenge of optimizing multiple agents in a sequential manner through the following principled approach:

- **Block-Coordinate Updates:** Agents are optimized one at a time while maintaining others in a frozen state, enabling focused optimization with reduced interference.
- **Distribution Correction:** Truncated Importance Sampling (IS) is employed to correct for distribution shift between the data collection policy and the current training policy.
- **Adaptive KL Control:** A quantile-based KL divergence controller ensures training stability by preventing excessive policy updates through adaptive constraint adjustment.

## Repository Structure
```
recipes/sat_seq/
├── README.md                   # Documentation
├── QUICK_START.md              # Getting started guide
├── __init__.py                 # Module initialization
├── real_controller.py          # Real-world data controller
├── agent_scheduler.py          # Agent update scheduler
├── stage_coordinator.py        # Training stage coordinator
├── train_dapo.sh               # DAPO training launcher
├── run_inference.sh            # Evaluation launcher
├── config/
│   └── sat_seq.yaml            # Hyperparameter configuration
├── adv/
│   └── seqaware.py             # Sequence-aware advantage estimation
├── loss/
│   └── seq_ratio_loss.py       # Policy ratio loss computation
├── kl/
│   └── quantile_kl_ctrl.py     # Quantile-based KL controller
├── data/
│   └── reweighting.py          # Importance sampling reweighting
└── monitor/
    └── certificate.py          # PAC certificate computation
```

## Quick Start

### Training with DAPO

To initiate training with Direct Advantage Policy Optimization:
```bash
bash recipes/sat_seq/train_dapo.sh
```

### Evaluation

The repository includes automated evaluation scripts for standard mathematical reasoning benchmarks.

**AIME 2024 Dataset:**
```bash
bash recipes/sat_seq/run_inference.sh --dataset aime24
```

**MATH Dataset:**
```bash
bash recipes/sat_seq/run_inference.sh --dataset math
```

**Custom Configuration:**
```bash
bash recipes/sat_seq/run_inference.sh \
    --dataset aime24 \
    --checkpoint_dir /path/to/checkpoints \
    --output_dir /path/to/results
```

## Algorithm Description

### Stage-Level Optimization Loop

1. **Data Collection:** Sample trajectories using the current policy π_cur to obtain on-policy data.

2. **Scheduling:** Determine the agent update sequence via the scheduler module.

3. **Sequential Agent Updates:** For each agent i in the determined order:
   - **Isolation:** Activate agent_i while freezing parameters of all other agents.
   - **Distribution Correction:** Apply importance sampling weights to account for policy distribution shift.
   - **Advantage Computation:** Calculate sequence-aware advantages using Generalized Advantage Estimation (GAE) with group-wise normalization.
   - **Optimization Iteration:**
     - Compute policy loss incorporating sequence ratio and KL penalty terms.
     - Perform gradient-based parameter update.
     - Evaluate KL divergence using quantile-based measurement.
     - **Constraint Enforcement:** If KL exceeds target threshold, revert update and increase penalty coefficient β; otherwise, accept update.

4. **Performance Certification:** Compute PAC-style lower bound on expected performance.

### Per-Agent Update Procedure

For each agent under optimization:

**Importance Weight Computation:**
```
c_t = min(1, exp(log π_hat(a_t|s_t) - log π_cur(a_t|s_t)))
```

**Advantage Processing:**
```
A_g = GAE(τ) → GroupNorm(A_g) → clip(A_g, ±A_clip)
```

**Objective Function:**
```
L = min{r_i · A_g, clip(r_i, 1-ε, 1+ε) · A_g} + β · KL(π_new || π_cur)
```

**Update Rule:**
- Compute gradients: ∇_θ L
- Apply optimizer step
- Evaluate: Q_{1-α}[KL(π_new || π_cur)] ≤ δ_target
  - If satisfied: Accept update
  - Otherwise: Revert parameters, update β ← 1.5β, retry (up to max_backtracks)

## Configuration

Core hyperparameters are specified in `config/sat_seq.yaml`:
```yaml
algorithm:
  gamma: 0.99              # Discount factor for return computation
  lam: 0.95                # GAE lambda parameter
  kl_ctrl:
    kl_coef: 0.001         # Initial KL penalty coefficient β
    target_kl: 0.05        # Target KL divergence threshold δ
    quantile: 0.99         # Quantile level for KL measurement (1-α)

sat_seq:
  epsilon: 0.2             # PPO clipping parameter ε
  A_clip: 5.0              # Advantage clipping threshold
  group_size: 4            # Responses per prompt for aggregation
  max_backtracks: 5        # Maximum backtracking iterations per update
  agents:
    - {name: agent_1, path: "path/to/model_1"}
    - {name: agent_2, path: "path/to/model_2"}
    - {name: agent_3, path: "path/to/model_3"}
```

## Data Format Specification

### Training Data

Training data should be formatted as JSONL with paired preference examples:
```json
{"prompt": "Problem statement", "chosen": "Preferred response", "rejected": "Dispreferred response"}
```

Example file location: `data/dapo/train.jsonl`

### Evaluation Data

Test sets should contain problems with ground-truth solutions:
```json
{"problem": "Problem description", "answer": "Reference solution"}
```

Example file location: `data/aime24/test.jsonl`

## Output Organization

### Training Artifacts

Model checkpoints and configuration files are saved to `outputs/sat_seq_dapo/`:
```
outputs/sat_seq_dapo/
├── train_config.yaml
├── agent_1_stage_10.pt
├── agent_2_stage_10.pt
└── agent_3_stage_10.pt
```

### Inference Results

Predictions are written to `outputs/aime24_inference/predictions.jsonl`:
```json
{
  "problem_id": 0,
  "predicted_answer": "28",
  "individual_answers": ["28", "28", "27"]
}
```

## Customization Guide

### Modifying Reward Functions

To implement custom reward logic, edit the `_compute_reward()` method in `real_controller.py`:
```python
import numpy as np

def _compute_reward(self, prompt: str, response: str) -> np.ndarray:
    """
    Custom reward computation based on response quality.
    
    Args:
        prompt: Input problem statement
        response: Generated solution
        
    Returns:
        Scalar reward value
    """
    if self._verify_correctness(response):
        return self._compute_quality_bonus(response)
    else:
        return self._compute_penalty(response)
```

### Adapting Evaluation Scripts

Evaluation parameters can be modified directly in `recipes/sat_seq/run_inference.sh`. Configurable elements include model paths, sampling temperature, generation parameters, and post-processing logic.

## Theoretical Foundations

### PAC Performance Certificate

The algorithm provides a lower bound on expected performance through:
```
L(π) = ∑_i (1-γ) · E[A_i] - Penalty_shift - Penalty_bias - Penalty_sample
```

where:
- **Information Gain:** ∑_i (1-γ) · E[A_i] quantifies expected improvement
- **Occupancy Shift:** Distribution mismatch penalty
- **Estimator Bias:** Approximation error in advantage estimation
- **Finite Sample Error:** Statistical uncertainty from limited data

### Robust KL Constraint

Rather than constraining the mean KL divergence, we enforce a quantile-based constraint for improved robustness:
```
Q_{1-α}[KL(π_target || π_current)] ≤ δ_target
```

This approach provides stronger guarantees by controlling tail behavior of the KL distribution, reducing sensitivity to outlier updates that could destabilize training.


## License

This project is released under the MIT License. See LICENSE file for details.

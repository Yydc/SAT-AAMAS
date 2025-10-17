SAT-Seq: Sequence-Aware Block-Coordinate Tuning
Note
This is an official recipe for the verl library. This directory should be placed within the verl/recipes/ folder to function correctly.
Introduction
SAT-Seq is a multi-agent reinforcement learning (MARL) algorithm designed to sequentially optimize multiple agents using a block-coordinate descent (BCD) approach. It provides PAC-style theoretical guarantees for its performance.
Core Idea: The algorithm updates one agent at a time while keeping others frozen. This process leverages truncated Importance Sampling (IS) to correct for distribution shift and employs a quantile-based KL controller to ensure stability during training.
Project Structure
code
Code
recipes/sat_seq/
├── README.md                   # This document
├── QUICK_START.md              # Quick start guide
├── 答案总结.md                  # Core Q&A (in Chinese)
├── __init__.py                 # Package entry point
├── real_controller.py          # Controller for real-world data (e.g., Qwen2-3B)
├── agent_scheduler.py          # Agent scheduler
├── stage_coordinator.py        # Stage coordinator
├── train_dapo.sh               # DAPO training script
├── run_inference.sh            # Evaluation script for AIME24/MATH
├── config/
│   └── sat_seq.yaml            # Configuration file
├── adv/
│   └── seqaware.py             # GAE for advantage estimation
├── loss/
│   └── seq_ratio_loss.py       # Sequence ratio loss
├── kl/
│   └── quantile_kl_ctrl.py     # Quantile-based KL control
├── data/
│   └── reweighting.py          # IS reweighting
└── monitor/
    └── certificate.py          # PAC certificate
Quick Start
1. DAPO Training
Execute the following command to start training with Direct Advantage Policy Optimization (DAPO):
code
Bash
bash recipes/sat_seq/train_dapo.sh
2. Evaluation on AIME24/MATH
Use the provided one-click evaluation script run_inference.sh.
Evaluate on the AIME24 dataset:
code
Bash
# Use the --dataset argument to specify the evaluation task
bash recipes/sat_seq/run_inference.sh --dataset aime24
Evaluate on the MATH dataset:
code
Bash
bash recipes/sat_seq/run_inference.sh --dataset math
Custom Parameters:
You can specify the model checkpoint and output directories.
code
Bash
# Point to your trained model checkpoints and a results directory
bash recipes/sat_seq/run_inference.sh \
    --dataset aime24 \
    --checkpoint_dir /path/to/your/checkpoints \
    --output_dir /path/to/your/results
Algorithm Workflow
Stage-Level Workflow
Collect on-policy data using the current policy (π_cur).
Determine the agent update order via the scheduler.
Sequentially update each agent:
a. Activate agent_i and freeze all other agents.
b. Apply Importance Sampling (IS) reweighting to correct for distribution shift.
c. Compute sequence-aware advantages (GAE + group-wise normalization).
d. Enter the optimization loop:
- Calculate the loss (sequence ratio + KL penalty).
- Perform a gradient update.
- Measure the KL divergence using a quantile-based approach.
- If KL > target: backtrack the update and increase the KL coefficient β.
- Else: accept the update.
Compute the PAC certificate (a lower bound on performance).
Per-Agent Update
For each agent_i:
Reweight: c_t = min(1, exp(logp_hat - logp_cur))
Advantage: A_g = GAE → Group-wise normalization → clip(±A_clip)
Loss: min{r_i * A_g, clip(r_i) * A_g} + β * KL
Update: loss.backward() and optimizer.step()
KL Check: Is Q_{0.99}[KL] ≤ δ_target?
Yes: Accept the update.
No: Backtrack, update β ← β * 1.5, and retry.
Configuration
Key parameters are defined in config/sat_seq.yaml.
code
Yaml
algorithm:
  gamma: 0.99              # Discount factor
  lam: 0.95                # GAE lambda
  kl_ctrl:
    kl_coef: 0.001         # Initial KL coefficient (β)
    target_kl: 0.05        # Target KL divergence
    quantile: 0.99         # KL quantile for robust control

sat_seq:
  epsilon: 0.2             # PPO clipping ratio
  A_clip: 5.0              # Advantage clipping
  group_size: 4            # Number of responses per prompt
  max_backtracks: 5        # Maximum number of backtracks per update
  agents:                  # Example with three Qwen2-3B models
    - {name: agent_1, path: "..."}
    - {name: agent_2, path: "..."}
    - {name: agent_3, path: "..."}
Data Format
DAPO Training Set (data/dapo/train.jsonl)
Each line is a JSON object with a prompt and chosen/rejected responses.
code
JSON
{"prompt": "Question text", "chosen": "Correct answer", "rejected": "Incorrect answer"}
AIME24 Test Set (data/aime24/test.jsonl)
Each line contains a problem and its ground-truth answer.
code
JSON
{"problem": "Problem description", "answer": "Correct answer"}
Output Structure
Training Outputs
Training artifacts are saved to outputs/sat_seq_dapo/:
code
Code
outputs/sat_seq_dapo/
├── train_config.yaml
├── agent_1_stage_10.pt
├── agent_2_stage_10.pt
└── agent_3_stage_10.pt
Inference Outputs
Predictions are saved to outputs/aime24_inference/predictions.jsonl:
code
Code
outputs/aime24_inference/
└── predictions.jsonl
Each line in predictions.jsonl has the following format:
code
JSON
{
  "problem_id": 0,
  "predicted_answer": "28",
  "individual_answers": ["28", "28", "27"]
}
Customization
Modifying the Reward Function
Edit the _compute_reward() method in real_controller.py:
code
Python
import numpy as np

def _compute_reward(self, prompt: str, response: str) -> np.ndarray:
    # Implement your custom reward logic here
    if is_answer_correct(response):
        return high_reward
    else:
        return low_reward
Modifying the Evaluation Script
All evaluation logic and parameters are centralized in recipes/sat_seq/run_inference.sh. You can directly modify this file to fit your needs, such as changing default model paths, sampling temperatures, or other generation parameters.
Theoretical Background
PAC Certificate
The performance lower bound is calculated as:
Lower Bound = Information Gain - Penalties
Where:
Information Gain = Σ_i (1-γ) * E[A_i]
Penalties = Occupancy Shift + Estimator Bias + Finite Sample Error
KL Constraint
Instead of using the mean KL divergence, we use a quantile-based constraint for greater robustness against outliers:
Q_{1-α}[KL(π_tar || π_cur)] ≤ δ_target
FAQ (Frequently Asked Questions)
Q: I'm getting a CUDA out of memory error.
A: Reduce the batch size in train_dapo.sh:
code
Bash
# In train_dapo.sh
TRAIN_BATCH_SIZE=256 # Decrease this value
Q: The model fails to load.
A: Ensure the model path is correct. You can use a Hugging Face model identifier to download it automatically:
code
Bash
# In train_dapo.sh
AGENT1_PATH="Qwen/Qwen2-7B-Instruct" # Change to a valid HF model
Q: Training is too slow.
A: Reduce the total number of training stages in train_dapo.sh:
code
Bash
# In train_dapo.sh
NUM_STAGES=2 # Decrease this value for faster, less thorough training
Further Reading
QUICK_START.md: A step-by-step guide to get started.
答案总结.md: Core Q&A about the DAPO training and AIME24 inference process (in Chinese).
config/sat_seq.yaml: Detailed explanations for all configuration parameters.
Citation
If you use this code in your research, please consider citing:
code
Bibtex
@misc{sat-seq-2025,
  title={SAT-Seq: Sequence-Aware Block-Coordinate Tuning},
  year={2025}
}
Get started: bash recipes/sat_seq/train_dapo.sh

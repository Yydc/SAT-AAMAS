#!/bin/bash
# SAT-Seq Training Script - Using the DAPO dataset
# 
# Usage:
#   bash recipe/sat_seq/train_dapo.sh
#
# Environment Requirements:
#   - PyTorch >= 2.0
#   - transformers >= 4.30
#   - CUDA (recommended)
#
# Dataset:
#   The DAPO dataset should be in JSONL format, with each line containing:
#   {"prompt": "question", "chosen": "correct_answer", "rejected": "incorrect_answer"}

set -e  # Exit immediately if a command exits with a non-zero status.

# ==================== Configuration ====================

# Project root directory
PROJECT_ROOT=$(cd "$(dirname "$0")/../.."; pwd)

# Dataset paths (math500 for training, aime24 for testing)
MATH500_TRAIN_PATH="${PROJECT_ROOT}/data/math500/train.jsonl"
AIME24_TEST_PATH="${PROJECT_ROOT}/data/aime24/test.jsonl"

# Model paths (three Qwen3-4B models, ensure these paths exist)
# If the models do not exist, you can use Hugging Face model names to download them automatically
AGENT1_PATH="${PROJECT_ROOT}/models/qwen3-4b-1"
AGENT2_PATH="${PROJECT_ROOT}/models/qwen3-4b-2"
AGENT3_PATH="${PROJECT_ROOT}/models/qwen3-4b-3"

# If you don't have local models, you can use models from Hugging Face
# Uncomment the lines below and comment out the local paths above
# AGENT1_PATH="Qwen/Qwen2.5-3B"
# AGENT2_PATH="Qwen/Qwen2.5-3B"  
# AGENT3_PATH="Qwen/Qwen2.5-3B"

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/outputs/sat_seq_dapo"

# Training hyperparameters
NUM_STAGES=10                    # Number of training stages
TRAIN_BATCH_SIZE=512            # Number of episodes per stage
MAX_RESPONSE_LENGTH=32768       # Max response length (consistent with experiment setup)
LEARNING_RATE=1e-5              # Learning rate

# SAT-Seq hyperparameters
EPSILON=0.2                     # PPO clip ratio
A_CLIP=5.0                      # Advantage clipping threshold
MAX_BACKTRACKS=5                # Max number of backtracks
KL_TARGET=0.05                  # KL target value
KL_COEF=0.001                   # KL coefficient
QUANTILE=0.99                   # KL quantile

# Scheduler mode: static, random, greedy_info_gain
SCHEDULER_MODE="static"

# GPU settings
export CUDA_VISIBLE_DEVICES=0   # Use the first GPU, can be set to 0,1,2,3 for multiple GPUs

# ==================== Environment Check ====================

echo "============================================"
echo "SAT-Seq Training with Math500 Dataset"
echo "============================================"

# Check for Python
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    exit 1
fi

# Check for necessary packages
python -c "import torch" 2>/dev/null || {
    echo "❌ Error: PyTorch not installed"
    echo "   Install with: pip install torch"
    exit 1
}

python -c "import transformers" 2>/dev/null || {
    echo "❌ Error: transformers not installed"
    echo "   Install with: pip install transformers"
    exit 1
}

python -c "import numpy" 2>/dev/null || {
    echo "❌ Error: numpy not installed"
    echo "   Install with: pip install numpy"
    exit 1
}

echo "✅ Environment check passed"

# Check for the training dataset
if [ ! -f "$MATH500_TRAIN_PATH" ]; then
    echo "⚠️  Warning: Math500 train set not found at $MATH500_TRAIN_PATH"
    echo "   Creating dummy Math500 train set for testing..."
    python -c "
import json
dummy_data = [
    {'prompt': 'Compute 2+2', 'chosen': '4', 'rejected': '5'},
    {'prompt': 'Compute 3*3', 'chosen': '9', 'rejected': '6'},
] * 100
import os
os.makedirs(os.path.dirname('$MATH500_TRAIN_PATH'), exist_ok=True)
with open('$MATH500_TRAIN_PATH', 'w') as f:
    for item in dummy_data:
        f.write(json.dumps(item) + '\n')
print('✅ Created dummy dataset with', len(dummy_data), 'examples')
"
fi

# Check for the test set (for subsequent evaluation/validation, does not affect training)
if [ ! -f "$AIME24_TEST_PATH" ]; then
    echo "⚠️  Warning: AIME24 test set not found at $AIME24_TEST_PATH"
    echo "   Creating dummy AIME24 test set for reference..."
    python -c "
import json, os
dummy_problems = [
    {'problem': 'AIME-style problem 1', 'answer': '28'},
    {'problem': 'AIME-style problem 2', 'answer': '13'},
] * 15
os.makedirs(os.path.dirname('$AIME24_TEST_PATH'), exist_ok=True)
with open('$AIME24_TEST_PATH', 'w') as f:
    for item in dummy_problems:
        f.write(json.dumps(item) + '\n')
print('✅ Created dummy AIME24 test set with', len(dummy_problems), 'problems')
"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==================== Create Training Configuration ====================

echo "Creating training configuration..."

CONFIG_FILE="${OUTPUT_DIR}/train_config.yaml"

cat > "$CONFIG_FILE" << EOF
# SAT-Seq Training Configuration for Math500
# Generated: $(date)

algorithm:
  gamma: 0.99
  lam: 0.95
  kl_penalty: kl
  kl_ctrl:
    type: per_state_quantile
    kl_coef: ${KL_COEF}
    target_kl: ${KL_TARGET}
    quantile: ${QUANTILE}

actor_rollout_ref:
  actor:
    use_kl_loss: true
    kl_loss_type: k1+
    kl_loss_coef: ${KL_COEF}
    ppo_epochs: 1
    clip_ratio: ${EPSILON}
    loss_agg_mode: token-mean
  rollout:
    n: 4
  ref: {}

data:
  train_batch_size: ${TRAIN_BATCH_SIZE}
  max_prompt_length: 512
  max_response_length: ${MAX_RESPONSE_LENGTH}

generation:
  temperature: 0.8
  top_p: 1.0

sat_seq:
  epsilon: ${EPSILON}
  A_clip: ${A_CLIP}
  group_size: 4
  adv_mode: grpo
  group_baseline: mean
  group_norm: true
  scheduler:
    mode: ${SCHEDULER_MODE}
  max_backtracks: ${MAX_BACKTRACKS}
  agents:
    - name: agent_1
      path: "${AGENT1_PATH}"
    - name: agent_2
      path: "${AGENT2_PATH}"
    - name: agent_3
      path: "${AGENT3_PATH}"

training:
  learning_rate: ${LEARNING_RATE}
  weight_decay: 0.01
  warmup_steps: 100
  vf_coef: 0.1
  clip_vf: true

logging:
  checkpoint_every_stage: 1
  every_n_steps: 10
  save_dir: ${OUTPUT_DIR}

dataset:
  train_path: ${MATH500_TRAIN_PATH}
  test_path: ${AIME24_TEST_PATH}
  type: math500
EOF

echo "✅ Configuration saved to $CONFIG_FILE"

# ==================== Create Training Script ====================

TRAIN_SCRIPT="${OUTPUT_DIR}/run_training.py"

cat > "$TRAIN_SCRIPT" << 'EOF'
"""
SAT-Seq Training Script - Using the Math500 training set.
"""

import sys
import argparse
from pathlib import Path

# Add project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import yaml
from recipe.sat_seq.agent_scheduler import AgentScheduler
from recipe.sat_seq.monitor.certificate import CertificateMonitor
from recipe.sat_seq.stage_coordinator import StageCoordinator

# Use the real controller
from recipe.sat_seq.real_controller import RealMultiAgentController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_stages", type=int, default=10)
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("SAT-Seq Training with Math500 Dataset")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Number of stages: {args.num_stages}")
    print(f"Train set: {config['dataset']['train_path']}")
    if 'test_path' in config.get('dataset', {}):
        print(f"Test set:  {config['dataset']['test_path']}")
    print("=" * 80)
    
    # Initialize components (using the real controller)
    print("\nInitializing components...")
    controller = RealMultiAgentController(
        config, 
        mode="train",
        dataset_path=config['dataset']['train_path']
    )
    
    scheduler_mode = config.get("sat_seq", {}).get("scheduler", {}).get("mode", "static")
    scheduler = AgentScheduler(mode=scheduler_mode, seed=42)
    
    gamma = config.get("algorithm", {}).get("gamma", 0.99)
    A_max = config.get("sat_seq", {}).get("A_clip", 5.0)
    monitor = CertificateMonitor(gamma=gamma, A_max=A_max, delta_conf=0.05)
    
    coordinator = StageCoordinator(config)
    
    print("✅ All components initialized\n")
    
    # Training loop
    stage_results = []
    for stage_idx in range(args.num_stages):
        print(f"\n{'='*80}")
        print(f"Stage {stage_idx + 1}/{args.num_stages}")
        print(f"{'='*80}\n")
        
        # Run one stage
        stage_result = coordinator.run_one_stage(controller, scheduler, monitor)
        stage_results.append(stage_result)
        
        # Print summary
        print(f"\n--- Stage {stage_idx + 1} Summary ---")
        print(f"  Lower Bound:           {stage_result['lower_bound']:.6f}")
        print(f"  Information Gain:      {stage_result['info_gain']:.6f}")
        print(f"  Penalties:             {stage_result['occ_shift_penalty']:.6f} + "
              f"{stage_result['estimator_bias_penalty']:.6f} + "
              f"{stage_result['finite_sample_penalty']:.6f}")
        
        # Save checkpoint
        if (stage_idx + 1) % config.get("logging", {}).get("checkpoint_every_stage", 1) == 0:
            save_dir = config.get("logging", {}).get("save_dir", "outputs/sat_seq")
            controller.save_checkpoint(save_dir, stage_idx + 1)
    
    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    total_lb = sum(r['lower_bound'] for r in stage_results)
    print(f"\nTotal Lower Bound: {total_lb:.6f}")
    print(f"Average per Stage: {total_lb / len(stage_results):.6f}")


if __name__ == "__main__":
    main()
EOF

echo "✅ Training script created at $TRAIN_SCRIPT"

# ==================== Start Training ====================

echo ""
echo "============================================"
echo "Starting Training..."
echo "============================================"
echo ""

cd "$PROJECT_ROOT"

python "$TRAIN_SCRIPT" \
    --config "$CONFIG_FILE" \
    --num_stages "$NUM_STAGES"

echo ""
echo "============================================"
echo "Training Completed!"
echo "============================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""


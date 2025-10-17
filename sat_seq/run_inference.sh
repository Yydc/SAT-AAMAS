#!/bin/bash
# SAT-Seq One-Click Inference Script - Supports AIME24 and MATH datasets
#
# Usage:
#   bash recipe/sat_seq/run_inference.sh --dataset aime24
#   bash recipe/sat_seq/run_inference.sh --dataset math --output_dir ./outputs/math_inference
#
# Arguments:
#   --dataset <name>      The dataset to evaluate on (aime24 or math) (required)
#   --checkpoint_dir <path> Directory containing the trained models (default: ./outputs/sat_seq_dapo)
#   --output_dir <path>     Directory to save the evaluation results (default: ./outputs/<dataset>_inference)
#   --num_samples <k>       Number of samples per agent for each problem (default: 64)

set -e

# ==================== Default Configuration ====================
PROJECT_ROOT=$(cd "$(dirname "$0")/../.."; pwd)
CHECKPOINT_DIR="${PROJECT_ROOT}/outputs/sat_seq_dapo"
NUM_SAMPLES_PER_AGENT=64
DATASET=""

# ==================== Parse Command-Line Arguments ====================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR_OVERRIDE="$2"; shift ;;
        --num_samples) NUM_SAMPLES_PER_AGENT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$DATASET" ]; then
    echo "Error: --dataset argument is required (e.g., aime24 or math)"
    exit 1
fi

# Set paths based on the dataset
if [ "$DATASET" == "aime24" ]; then
    DATASET_PATH="${PROJECT_ROOT}/data/aime24/test.jsonl"
    DATASET_TYPE="aime24"
elif [ "$DATASET" == "math" ]; then
    DATASET_PATH="${PROJECT_ROOT}/data/math/test.jsonl"
    DATASET_TYPE="math"
else
    echo "Error: Unsupported dataset '$DATASET'. Please choose 'aime24' or 'math'."
    exit 1
fi

# Set output directory
if [ -n "$OUTPUT_DIR_OVERRIDE" ]; then
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
else
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/${DATASET}_inference"
fi
PREDICTIONS_FILE="${OUTPUT_DIR}/predictions.jsonl"

# ==================== Configure Model Paths ====================
AGENT1_PATH="${PROJECT_ROOT}/models/qwen3-4b-1"
AGENT2_PATH="${PROJECT_ROOT}/models/qwen3-4b-2"
AGENT3_PATH="${PROJECT_ROOT}/models/qwen3-4b-3"
AGENT1_CHECKPOINT="${CHECKPOINT_DIR}/agent_1_stage_10.pt"
AGENT2_CHECKPOINT="${CHECKPOINT_DIR}/agent_2_stage_10.pt"
AGENT3_CHECKPOINT="${CHECKPOINT_DIR}/agent_3_stage_10.pt"


# ==================== Main Script Body ====================

echo "============================================"
echo "SAT-Seq Inference on ${DATASET^^}"
echo "============================================"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Dataset path: ${DATASET_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================"

# Environment check
python -c "import torch; import transformers; import numpy" 2>/dev/null || {
    echo "❌ Error: Required packages not installed. Install with: pip install torch transformers numpy"
    exit 1
}
echo "✅ Environment check passed"

# Check for dataset
if [ ! -f "$DATASET_PATH" ]; then
    echo "⚠️  Warning: ${DATASET^^} test set not found at: $DATASET_PATH"
    echo "   Creating a dummy dataset for demonstration..."
    mkdir -p "$(dirname "$DATASET_PATH")"
    if [ "$DATASET" == "aime24" ]; then
        python -c "import json; f=open('$DATASET_PATH','w'); [f.write(json.dumps({'problem': f'AIME problem {i}', 'answer': str(i*i)})+'\n') for i in range(10)]"
    else # math
        python -c "import json; f=open('$DATASET_PATH','w'); [f.write(json.dumps({'prompt': f'Solve {i}+{i}', 'chosen': str(2*i)})+'\n') for i in range(10)]"
    fi
    echo "✅ Dummy dataset created"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==================== Create Inference Configuration ====================
CONFIG_FILE="${OUTPUT_DIR}/inference_config.yaml"
cat > "$CONFIG_FILE" << EOF
# SAT-Seq Inference Configuration for ${DATASET^^}
# Generated: $(date)
data:
  max_response_length: 32768
sat_seq:
  agents:
    - {name: agent_1, path: "${AGENT1_PATH}", checkpoint: "${AGENT1_CHECKPOINT}"}
    - {name: agent_2, path: "${AGENT2_PATH}", checkpoint: "${AGENT2_CHECKPOINT}"}
    - {name: agent_3, path: "${AGENT3_PATH}", checkpoint: "${AGENT3_CHECKPOINT}"}
inference:
  temperature: 0.8
  top_p: 1.0
  num_samples_per_agent: ${NUM_SAMPLES_PER_AGENT}
  ensemble_strategy: "majority_vote"
dataset:
  test_path: ${DATASET_PATH}
  type: ${DATASET_TYPE}
output:
  predictions_file: ${PREDICTIONS_FILE}
EOF
echo "✅ Inference configuration saved to $CONFIG_FILE"

# ==================== Create Python Inference Script ====================
INFERENCE_SCRIPT="${OUTPUT_DIR}/run_inference.py"
cat > "$INFERENCE_SCRIPT" << 'EOF'
"""
SAT-Seq Generic Inference Script
"""
import sys, json, argparse, re
from pathlib import Path
from typing import List, Dict
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import yaml
import numpy as np
try:
    import torch
    from recipe.sat_seq.real_controller import RealMultiAgentController
except ImportError:
    print("⚠️ PyTorch or project modules not available.")
    sys.exit(1)

def extract_answer(response: str) -> str:
    patterns = [r"\\boxed\{(.+?)\}", r"(?:answer|is|is:)\s*([0-9,./]+)"]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match: return match.group(1).strip()
    numbers = re.findall(r"[-+]?\d+(?:/\d+)?", response)
    return numbers[-1] if numbers else "no_answer"

def normalize_number(s: str) -> str:
    from fractions import Fraction
    t = s.strip().replace(",", "").split('.')[0]
    return str(Fraction(t)) if "/" in t else str(int(t))

def verify_math(pred: str, truth: str) -> bool:
    if not isinstance(truth, str) or not truth.strip() or pred == "no_answer": return False
    try: return normalize_number(pred) == normalize_number(truth)
    except Exception: return pred.strip() == truth.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    dataset_type = config['dataset']['type']
    
    print("=" * 80)
    print(f"SAT-Seq Inference on {dataset_type.upper()}")
    print(f"Test set: {config['dataset']['test_path']}")
    print("=" * 80)
    
    controller = RealMultiAgentController(config, mode="inference", dataset_path=config['dataset']['test_path'])
    for i, agent_cfg in enumerate(config['sat_seq']['agents']):
        ckpt_path = agent_cfg.get('checkpoint')
        if ckpt_path and Path(ckpt_path).exists():
            print(f"Loading checkpoint for {agent_cfg['name']} from {ckpt_path}...")
            controller.agents[i]['model'].load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    
    test_problems, predictions = [], []
    with open(config['dataset']['test_path'], 'r') as f:
        test_problems = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(test_problems)} test problems\n")

    for i, problem_data in enumerate(test_problems):
        prompt = problem_data.get("problem") or problem_data.get("prompt", "")
        print(f"\n[{i+1}/{len(test_problems)}] Processing: {prompt[:80]}...")
        
        all_answers = []
        for agent in controller.agents:
            if agent['model'] is None: continue
            for _ in range(config['inference']['num_samples_per_agent']):
                response, _, _, _, _, _ = controller._generate_single_response(
                    agent, prompt, config['data']['max_response_length']
                )
                all_answers.append(extract_answer(response))

        truth = problem_data.get("answer") or problem_data.get("chosen", "")
        correctness = [verify_math(a, truth) for a in all_answers]
        pass_at_k = any(correctness)
        avg_at_k = np.mean(correctness) if correctness else 0.0

        predictions.append({
            "problem_id": i, "prompt": prompt, "pass_at_k": bool(pass_at_k), "avg_at_k": avg_at_k,
        })
    
    with open(config['output']['predictions_file'], 'w') as f:
        for p in predictions: f.write(json.dumps(p) + '\n')
    
    print(f"\n✅ Predictions saved to {config['output']['predictions_file']}")
    
    if all(p.get("pass_at_k") is not None for p in predictions):
        pass_rate = np.mean([p['pass_at_k'] for p in predictions]) * 100
        avg_k_rate = np.mean([p['avg_at_k'] for p in predictions]) * 100
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"pass@K: {pass_rate:.2f}%")
        print(f"avg@K:  {avg_k_rate:.2f}%")
        print("=" * 80)

if __name__ == "__main__":
    main()
EOF
echo "✅ Python inference script created"

# ==================== Start Inference ====================
echo -e "\n============================================"
echo "Starting Inference..."
echo "============================================"
cd "$PROJECT_ROOT"
python "$INFERENCE_SCRIPT" --config "$CONFIG_FILE"
echo -e "\n============================================"
echo "Inference Completed!"
echo "============================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""

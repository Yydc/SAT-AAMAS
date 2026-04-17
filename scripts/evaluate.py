"""SAT evaluation entry point (avg@K / pass@K).

Usage:
    python scripts/evaluate.py --config configs/sat_demo.yaml
    python scripts/evaluate.py --config configs/sat_default.yaml --ckpt_dir outputs/sat_default
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from sat.real_controller import RealMultiAgentController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SAT team.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Directory with agent_*_stage_*.pt checkpoints.")
    parser.add_argument("--stage", type=int, default=None,
                        help="Stage index to load; defaults to the latest available.")
    parser.add_argument("--output", type=str, default=None,
                        help="Predictions jsonl path (default: <save_dir>/predictions.jsonl).")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def extract_answer(text: str) -> str:
    for pattern in (r"\\boxed\{([^{}]+)\}", r"(?:answer|result)[\s:=]+([-+]?\d+(?:/\d+)?)"):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    nums = re.findall(r"[-+]?\d+(?:/\d+)?", text)
    return nums[-1] if nums else ""


def normalise(value: str) -> str:
    value = value.strip().replace(",", "")
    if "/" in value:
        return str(Fraction(value))
    try:
        return str(int(value))
    except ValueError:
        return value


def is_match(pred: str, truth: str) -> bool:
    if not pred or not truth:
        return False
    try:
        return normalise(pred) == normalise(truth)
    except Exception:
        return pred.strip() == truth.strip()


def find_latest_checkpoints(ckpt_dir: Path, stage: int | None) -> dict[str, Path]:
    if not ckpt_dir.exists():
        return {}
    mapping: dict[str, list[tuple[int, Path]]] = {}
    for path in ckpt_dir.glob("*_stage_*.pt"):
        name, _, tail = path.stem.rpartition("_stage_")
        try:
            stage_idx = int(tail)
        except ValueError:
            continue
        mapping.setdefault(name, []).append((stage_idx, path))
    resolved = {}
    for name, items in mapping.items():
        if stage is not None:
            match = [p for s, p in items if s == stage]
            if match:
                resolved[name] = match[0]
        else:
            resolved[name] = max(items, key=lambda sp: sp[0])[1]
    return resolved


def iter_problems(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    test_path = Path(cfg["dataset"]["test_path"])
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test set not found: {test_path}. "
            "Run `python scripts/prepare_data.py --dataset <name>` first."
        )

    save_dir = Path(cfg.get("logging", {}).get("save_dir", "outputs/sat"))
    output_path = Path(args.output) if args.output else save_dir / "predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    controller = RealMultiAgentController(cfg, mode="inference", dataset_path=str(test_path))

    if args.ckpt_dir:
        ckpts = find_latest_checkpoints(Path(args.ckpt_dir), args.stage)
        if not ckpts:
            print(f"[warn] no checkpoints found under {args.ckpt_dir}")
        else:
            import torch
            for agent in controller.agents:
                key = agent.get("name")
                if agent["model"] is None or key not in ckpts:
                    continue
                state = torch.load(ckpts[key], map_location="cpu")
                agent["model"].load_state_dict(state["model_state_dict"])
                print(f"[ckpt] {key} <- {ckpts[key]}")

    num_samples = cfg.get("evaluation", {}).get("num_samples_per_agent", 4)
    max_len = cfg["data"]["max_response_length"]

    predictions = []
    for idx, problem in enumerate(iter_problems(test_path)):
        prompt = problem.get("problem") or problem.get("prompt", "")
        truth = problem.get("answer") or problem.get("chosen", "")
        answers = []
        for agent in controller.agents:
            if agent["model"] is None:
                continue
            for _ in range(num_samples):
                response, *_ = controller._generate_single_response(agent, prompt, max_len)
                answers.append(extract_answer(response))
        correctness = [is_match(a, truth) for a in answers] if truth else []
        pass_at_k = bool(any(correctness)) if correctness else None
        avg_at_k = float(np.mean(correctness)) if correctness else None
        predictions.append({
            "problem_id": idx,
            "prompt": prompt,
            "answers": answers,
            "pass_at_k": pass_at_k,
            "avg_at_k": avg_at_k,
        })
        print(f"[{idx + 1}] pass@K={pass_at_k} avg@K={avg_at_k}")

    with open(output_path, "w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row) + "\n")
    print(f"\nPredictions -> {output_path}")

    scored = [p for p in predictions if p["pass_at_k"] is not None]
    if scored:
        pass_rate = 100.0 * np.mean([p["pass_at_k"] for p in scored])
        avg_rate = 100.0 * np.mean([p["avg_at_k"] for p in scored])
        print("=" * 80)
        print(f"pass@K: {pass_rate:.2f}%")
        print(f"avg@K : {avg_rate:.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    main()

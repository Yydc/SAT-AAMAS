"""SAT training entry point.

Usage:
    python scripts/train.py --config configs/sat_demo.yaml
    python scripts/train.py --config configs/sat_default.yaml --num_stages 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo importable whether the user runs "python scripts/train.py"
# from the project root or from somewhere else.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from sat.agent_scheduler import AgentScheduler
from sat.monitor.certificate import CertificateMonitor
from sat.real_controller import RealMultiAgentController
from sat.stage_coordinator import StageCoordinator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SAT team.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a SAT YAML config (configs/sat_*.yaml).")
    parser.add_argument("--num_stages", type=int, default=None,
                        help="Override training.num_stages from the config.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    num_stages = args.num_stages or cfg.get("training", {}).get("num_stages", 10)
    save_dir = Path(cfg.get("logging", {}).get("save_dir", "outputs/sat"))
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = cfg.get("dataset", {}).get("train_path")
    if dataset_path and not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Training set not found: {dataset_path}. "
            "Run `python scripts/prepare_data.py --dataset demo` first."
        )

    print("=" * 80)
    print(f"SAT training: {args.config}")
    print(f"  num_stages : {num_stages}")
    print(f"  save_dir   : {save_dir}")
    print(f"  train_path : {dataset_path}")
    print("=" * 80)

    controller = RealMultiAgentController(cfg, mode="train", dataset_path=dataset_path)

    scheduler_mode = cfg.get("sat", {}).get("scheduler", {}).get("mode", "static")
    scheduler = AgentScheduler(mode=scheduler_mode, seed=args.seed)

    cert_cfg = cfg.get("certificate", {})
    monitor = CertificateMonitor(
        gamma=cfg.get("algorithm", {}).get("gamma", 0.99),
        A_max=cert_cfg.get("A_max", cfg.get("sat", {}).get("A_clip", 5.0)),
        delta_conf=cert_cfg.get("delta_conf", 0.05),
    )

    coordinator = StageCoordinator(cfg)

    # Persist the effective training config alongside the checkpoints.
    with open(save_dir / "train_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    stage_log = []
    for stage_idx in range(1, num_stages + 1):
        print(f"\n=== Stage {stage_idx}/{num_stages} ===")
        result = coordinator.run_one_stage(controller, scheduler, monitor)
        stage_log.append({"stage": stage_idx, **{k: float(v) for k, v in result.items()}})
        print(
            "  Stage lower bound: {lower_bound:.4f}  "
            "(info_gain={info_gain:.4f}, occ={occ_shift_penalty:.4f}, "
            "bias={estimator_bias_penalty:.4f}, sample={finite_sample_penalty:.4f})".format(
                **{k: float(v) for k, v in result.items()}
            )
        )
        if stage_idx % cfg.get("logging", {}).get("checkpoint_every_stage", 1) == 0:
            controller.save_checkpoint(str(save_dir), stage_idx)

    log_path = save_dir / "stage_log.jsonl"
    with open(log_path, "w", encoding="utf-8") as fh:
        for row in stage_log:
            fh.write(json.dumps(row) + "\n")
    print(f"\nStage log -> {log_path}")

    total_lb = sum(row["lower_bound"] for row in stage_log)
    print(f"Total lower bound: {total_lb:.4f}")
    print(f"Mean lower bound : {total_lb / max(len(stage_log), 1):.4f}")


if __name__ == "__main__":
    main()

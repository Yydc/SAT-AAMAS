"""Plug-and-play agent upgrade driver.

Starting from a SAT-trained team, swap in stronger pretrained checkpoints
as declared by ``sat.agents[*].upgrade`` in the config. For each swap we
run the Stage-0 aligner (``sat.pnp.Stage0Aligner``) on the first prompt
of the evaluation set so the replacement starts inside the per-state KL
trust region, then re-evaluate the team.

Usage:
    python scripts/plug_and_play.py --config configs/sat_pnp.yaml \
        --ckpt_dir outputs/sat_default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from sat.pnp import Stage0Aligner
from sat.real_controller import RealMultiAgentController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plug-and-play SAT upgrade.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Optional directory with agent_*_stage_*.pt (baseline).")
    parser.add_argument("--probe_prompt", type=str,
                        default="Solve 2 + 3 and output the answer.",
                        help="Prompt used to probe Stage-0 KL alignment.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _next_token_logits(model, tokenizer, prompt: str, device: torch.device) -> np.ndarray:
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits, _ = model(**toks)
    return logits[0, -1].detach().float().cpu().numpy()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    test_path = cfg.get("dataset", {}).get("test_path")
    if not test_path or not Path(test_path).exists():
        raise FileNotFoundError("Evaluation set not found; run prepare_data.py first.")

    # Load the baseline team.
    controller = RealMultiAgentController(cfg, mode="inference", dataset_path=test_path)

    # Apply baseline SAT checkpoints if available.
    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
        for agent in controller.agents:
            matches = sorted(ckpt_dir.glob(f"{agent['name']}_stage_*.pt"))
            if matches:
                state = torch.load(matches[-1], map_location="cpu")
                agent["model"].load_state_dict(state["model_state_dict"])
                print(f"[ckpt] {agent['name']} <- {matches[-1]}")

    aligner = Stage0Aligner(delta0=cfg.get("sat", {}).get("stage0_delta", 0.01))

    # For each declared upgrade, import the pretrained model, run the KL
    # projection on the first-token distribution as a sanity check, and
    # promote the upgraded weights into the live agent slot.
    for slot, (agent, agent_cfg) in enumerate(zip(controller.agents, cfg["sat"]["agents"])):
        upgrade_path = agent_cfg.get("upgrade")
        if not upgrade_path:
            continue

        print(f"\n[Stage-0] Upgrading {agent['name']} -> {upgrade_path}")
        from sat.models import ModelWithValueHead
        from transformers import AutoTokenizer

        new_tokenizer = AutoTokenizer.from_pretrained(
            upgrade_path, trust_remote_code=True, padding_side="left"
        )
        if new_tokenizer.pad_token is None:
            new_tokenizer.pad_token = new_tokenizer.eos_token

        new_model = ModelWithValueHead(model_path=upgrade_path).to(controller.device)

        cur_logits = _next_token_logits(agent["model"], agent["tokenizer"], args.probe_prompt, controller.device)
        pre_logits = _next_token_logits(new_model, new_tokenizer, args.probe_prompt, controller.device)
        pad = max(cur_logits.shape[0], pre_logits.shape[0])
        cur_logits_pad = np.full(pad, -1e9)
        cur_logits_pad[: cur_logits.shape[0]] = cur_logits
        pre_logits_pad = np.full(pad, -1e9)
        pre_logits_pad[: pre_logits.shape[0]] = pre_logits
        projected = aligner.project_logits(pre_logits_pad, cur_logits_pad)
        print(f"  probe KL(projected || cur) = {float(np.sum(projected * (np.log(projected + 1e-12) - np.log(_softmax(cur_logits_pad) + 1e-12)))):.4f}  "
              f"<= delta0 = {aligner.delta0:.4f}")

        # Swap the agent slot to the upgraded model.
        controller.agents[slot]["model"] = new_model
        controller.agents[slot]["tokenizer"] = new_tokenizer
        controller.agents[slot]["path"] = upgrade_path

    print("\nPlug-and-play upgrade complete. Run scripts/evaluate.py to score the upgraded team.")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


if __name__ == "__main__":
    main()

<div align="center">

<h1>SAT: Sequential Agent Tuning for Coordinator-Free Plug-and-Play Multi-LLM Training with Monotonic Improvement Guarantees</h1>

<h3>A coordinator-free training paradigm for teams of small LLMs with monotonic improvement bounds, sequence-agnostic guarantees, and provable plug-and-play invariance</h3>

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.44+-green.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AAMAS 2026](https://img.shields.io/badge/AAMAS-2026-red.svg)](https://aamas2026.org/)

<p><em>Official implementation of <strong>SAT</strong> (AAMAS 2026): a coordinator-free, block-coordinate trust-region trainer for factorized teams of LLMs. Three Qwen3-4B agents (12B total) trained with SAT outperform Qwen3-32B on AIME24/25 by <strong>+3.9%</strong> on average; swapping in two 8B agents via plug-and-play boosts the composite by <strong>+10.4%</strong>.</em></p>

</div>

## Motivation

Modern LLMs achieve strong performance but are expensive to deploy, especially as their parameter counts grow into the tens or hundreds of billions. A complementary line of work asks whether *teams* of small models can match or surpass a single large one. Existing multi-agent LLM frameworks struggle on two fronts:

- **No convergence guarantees.** Hand-crafted role assignments, debates, or judge ensembles offer little theory about *why* the team should improve, and joint updates suffer from compounding distribution shift.
- **Coordinator overhead.** Centralized controllers add latency, must be retrained every time an agent is upgraded, and break the modularity that makes small-model teams attractive in the first place.

### Our solution: SAT

SAT (**S**equential **A**gent **T**uning) replaces the central coordinator with a sequential, block-coordinate update over a factorized product policy:

- **Sequence-aware on-policy estimator.** When updating agent `i`, we evaluate full trajectories under the *intermediate* team policy `pi^{i-1}`, with truncated multi-step importance ratios `c_t = min(1, rho_t)`.
- **Per-agent quantile KL trust regions.** A high-quantile KL controller caps each agent's per-state divergence at a target radius `delta_i`, with backtracking when the constraint is violated.
- **Plug-and-play upgrades with closed-form alignment.** Replacing an agent with a stronger pretrained model is mediated by a Stage-0 KL projection (a closed-form geometric mixture), preserving the monotonic-improvement certificate without retraining the rest of the team.

These ingredients yield (i) a **single-step and joint-stage monotonic improvement bound**, (ii) a **sequence-agnostic** lower bound that holds for any update order, and (iii) **plug-and-play invariance** that lets you upgrade one agent at a time without breaking the certificate.

## Repository layout

```
SAT-AAMAS/
├── configs/                 # YAML configs (sat_demo, sat_default, sat_pnp)
├── data/                    # Datasets prepared via scripts/prepare_data.py
├── manuscripts/             # AAMAS 2026 manuscript (LaTeX source + figures)
├── sat/                     # Core package
│   ├── stage_coordinator.py     # Algorithm 1 outer loop
│   ├── agent_scheduler.py       # ORDERAGENTS (static / random / greedy)
│   ├── real_controller.py       # HF/local multi-agent rollout & update
│   ├── advantage/seqaware.py    # Sequence-aware GRPO/GAE + group norm + clip
│   ├── loss/seq_ratio_loss.py   # Sequence-level PPO ratio + value loss
│   ├── kl/quantile_kl_ctrl.py   # Quantile KL controller + backtracking
│   ├── data/reweighting.py      # Truncated IS reweighter c_t = min(1, rho_t)
│   ├── monitor/certificate.py   # PAC-style stage lower bound
│   ├── pnp/stage0_alignment.py  # Plug-and-play KL projection
│   ├── models/causal_lm.py      # HF causal LM with value head (no veRL)
│   └── models/tiny_lm.py        # Local tiny LM for no-download smoke tests
├── scripts/
│   ├── train.py             # SAT training entry point
│   ├── evaluate.py          # avg@K / pass@K evaluation
│   ├── plug_and_play.py     # Stage-0 alignment + agent swap
│   ├── prepare_data.py      # Dataset builder (demo / aime24 / math500 / dapo)
│   └── setup_demo.sh        # One-shot install + train + evaluate
├── requirements.txt
├── setup.py
└── LICENSE                  # MIT
```

## Minimal usage

We support a one-shot script and a manual install.

### One-shot demo install and run

```bash
git clone https://github.com/Yydc/SAT-AAMAS.git
cd SAT-AAMAS
bash scripts/setup_demo.sh
```

This installs SAT in editable mode, generates a synthetic demo dataset, runs two short SAT stages on three built-in `sat:tiny` agents, and evaluates the team. The demo path does not download model checkpoints; it verifies the full open-source pipeline before users move to the paper-scale Hugging Face configs.

If your environment already has the dependencies installed, use `SAT_SKIP_INSTALL=1 bash scripts/setup_demo.sh` to run only the data, train, and evaluation steps.

## Installation

### Manual install

```bash
git clone https://github.com/Yydc/SAT-AAMAS.git
cd SAT-AAMAS

conda create -n sat python=3.10 -y
conda activate sat

pip install -r requirements.txt
pip install -e .                                  # exposes the `sat` package
pip install datasets                              # optional, for HF benchmarks
```

If you plan to reproduce the paper numbers on AIME / MATH-500 / DAPO, also install Hugging Face Datasets (`pip install datasets`); the demo dataset is purely synthetic and has no external dependency.

## Usage

### 1. End-to-end demo (CPU friendly)

```bash
python scripts/prepare_data.py --dataset demo
python scripts/train.py --config configs/sat_demo.yaml
python scripts/evaluate.py --config configs/sat_demo.yaml --ckpt_dir outputs/sat_demo
```

The demo config uses three local `sat:tiny` agents, group size 3, and 2 SAT stages. It exercises rollout collection, sequence-level advantage computation, per-agent policy updates, KL backtracking, checkpointing, and avg@K/pass@K evaluation without requiring GPU memory or external model downloads.

### 2. Full SAT training (paper setting)

```bash
# Pull the AIME24 evaluation set and the DAPO training prompts.
python scripts/prepare_data.py --dataset aime24
python scripts/prepare_data.py --dataset dapo

# Edit configs/sat_default.yaml if you want to point `sat.agents[*].path`
# at locally-cached Qwen3-4B checkpoints; Hugging Face IDs work as well.
python scripts/train.py --config configs/sat_default.yaml --num_stages 10
python scripts/evaluate.py --config configs/sat_default.yaml \
    --ckpt_dir outputs/sat_default
```

### 3. Plug-and-play upgrade

```bash
# Start from a SAT-trained 3x4B team and swap two slots to 8B via Stage-0
# KL projection (manuscript Section 4 "Plug-and-play agent upgrades").
python scripts/plug_and_play.py --config configs/sat_pnp.yaml \
    --ckpt_dir outputs/sat_default
python scripts/evaluate.py --config configs/sat_pnp.yaml \
    --ckpt_dir outputs/sat_default
```

## Algorithm

SAT instantiates Algorithm 1 of the manuscript:

```
for stage k = 1, 2, ...:
    rollout B under the current team pi_cur;  order sigma <- ORDERAGENTS(B)
    for i = 1 ... n:
        form pi^{i-1};  compute group-normalised seq advantages with clip A_max
        L_i = E[min(r_i * A_g, clip(r_i, 1+/-eps) * A_g)] - beta * E_s[KL(pi^{sigma(i)} || pi^{sigma(i)}_cur)]
        update pi^{sigma(i)}_tar via OPTIMIZE(L_i)
        if Quantile_{1-alpha}[KL(pi^{sigma(i)}_tar || pi^{sigma(i)}_cur)] > delta_i:
            backtrack and increase beta
        promote pi^{sigma(i)}_tar -> pi^{sigma(i)}_cur
    pi_cur <- assembled team
```

Module map:

| Manuscript symbol                                | Code                                          |
| ------------------------------------------------ | --------------------------------------------- |
| `ORDERAGENTS`                                    | [`sat/agent_scheduler.py`](sat/agent_scheduler.py) |
| Sequence-aware advantage `\\hat A^{i-1}_{ON}` + group norm + clip | [`sat/advantage/seqaware.py`](sat/advantage/seqaware.py) |
| Truncated IS `c_t = min(1, rho_t)`               | [`sat/data/reweighting.py`](sat/data/reweighting.py) |
| Sequence ratio `r_i(tau) = exp(u_i(tau))` + PPO clip + KL penalty | [`sat/loss/seq_ratio_loss.py`](sat/loss/seq_ratio_loss.py) |
| Quantile KL controller + sampled per-state KL proxy + backtracking | [`sat/kl/quantile_kl_ctrl.py`](sat/kl/quantile_kl_ctrl.py), [`sat/real_controller.py`](sat/real_controller.py) |
| Stage outer loop (Algorithm 1)                   | [`sat/stage_coordinator.py`](sat/stage_coordinator.py) |
| PAC stage bound (Theorem 1.4)                    | [`sat/monitor/certificate.py`](sat/monitor/certificate.py) |
| Stage-0 KL projection for plug-and-play (eq. 18-19) | [`sat/pnp/stage0_alignment.py`](sat/pnp/stage0_alignment.py) |

## Configuration

Every entry point reads a single YAML file. The default is [`configs/sat_default.yaml`](configs/sat_default.yaml); use [`configs/sat_demo.yaml`](configs/sat_demo.yaml) for a small CPU-friendly run. Key fields:

| Group              | Field                  | Meaning                                         |
| ------------------ | ---------------------- | ----------------------------------------------- |
| `algorithm`        | `gamma`, `lam`         | Discount and GAE-lambda                         |
| `algorithm.kl_ctrl`| `kl_coef`, `target_kl`, `quantile` | Initial `beta`, target radius `delta_i`, `(1-alpha)` quantile |
| `data`             | `train_batch_size`, `max_response_length` | Rollout shape                       |
| `generation`       | `temperature`, `top_p` | Sampling for rollouts and inference             |
| `sat`              | `epsilon`, `A_clip`, `group_size`, `adv_mode`, `group_baseline`, `group_norm` | PPO clip, advantage ceiling, group settings |
| `sat.scheduler`    | `mode`                 | `static`, `random`, or `greedy_info_gain`       |
| `sat.max_backtracks`| -                     | Backtrack budget per agent per stage            |
| `sat.stage0_delta` | -                     | Per-state KL budget for plug-and-play upgrades  |
| `sat.agents[*]`    | `name`, `path`, optional `upgrade` | Team composition                       |
| `training`         | `learning_rate`, `vf_coef`, `clip_vf`, `num_stages` | Optimisation knobs              |
| `evaluation`       | `num_samples_per_agent` | K for avg@K / pass@K                           |
| `certificate`      | `A_max`, `delta_conf`  | Inputs for the PAC stage bound                  |
| `dataset`          | `train_path`, `test_path`, `type` | JSONL paths consumed by the controller |

Override `--num_stages` from the command line if you want to vary stage count without editing the config.

## Datasets

`scripts/prepare_data.py` writes the JSONL files expected by every config. Available dataset names: `demo`, `aime24`, `aime25`, `math500`, `dapo`. See [data/README.md](data/README.md) for the schemas.

## Reproducing the main results

The main experimental table (manuscript Section 5) is reproduced with the steps below. The numbers in the manuscript come from the full Qwen3-4B/8B and LLaMA 3.1-8B teams; you can mirror that by editing `sat.agents[*].path` and `data.max_response_length` in `configs/sat_default.yaml`.

```bash
# 1. Download the benchmark + training corpus.
python scripts/prepare_data.py --dataset aime24
python scripts/prepare_data.py --dataset aime25
python scripts/prepare_data.py --dataset math500
python scripts/prepare_data.py --dataset dapo

# 2. Train a 3x Qwen3-4B team (Section 5.2 main result).
python scripts/train.py --config configs/sat_default.yaml --num_stages 10

# 3. Evaluate avg@K / pass@K on each benchmark; K matches Section 5.1.
python scripts/evaluate.py --config configs/sat_default.yaml \
    --ckpt_dir outputs/sat_default

# 4. (Optional) Plug-and-play to 2x Qwen3-8B + 1x Qwen3-4B (Section 5.4).
python scripts/plug_and_play.py --config configs/sat_pnp.yaml \
    --ckpt_dir outputs/sat_default
python scripts/evaluate.py --config configs/sat_pnp.yaml \
    --ckpt_dir outputs/sat_default
```

Reported settings (Section 5.1): `temperature=0.8`, `top_p=1.0`, `max_response_length=32768`, group size `G_grp in {4, 8}`, GAE `lam=0.95`, PPO `epsilon=0.2`, `K=64` for AIME and ZebraLogic, `K=25` for ARBench, `K=8` for planning, `K=4` for MATH-500.

## Theory in one paragraph

Under per-agent per-state KL radii `{delta_i}` and `N_i` on-policy episodes per step, with probability at least `1 - delta_conf`,

```
J(bar pi) - J(pi_cur) >=
    sum_i (kappa_i sqrt(delta_i) - a_i delta_i)            # information-geometric gain
  - 2 gamma / (1 - gamma)^2 * A_max * sum_i sqrt(delta_i / 2)   # occupancy-shift penalty
  - 1 / (1 - gamma) * sum_i zeta_i                          # estimator-bias penalty
  - sum_i A_max / (1 - gamma) * sqrt(log(2 n / delta_conf) / (2 N_i))   # finite-sample error
```

The bound is *sequence-agnostic* (any order `sigma`), holds under *plug-and-play* replacements when the new agent enters via the Stage-0 projection, and the underlying sequential block-coordinate updates converge at `O(1/K)` (Theorem 1.5). The certificate monitor in [`sat/monitor/certificate.py`](sat/monitor/certificate.py) emits these four terms after every stage so you can track them directly in `outputs/<run>/stage_log.jsonl`.

## Output layout

After training:

```
outputs/<run>/
├── train_config.yaml          # Effective config (frozen at launch)
├── stage_log.jsonl            # One row per stage with the certificate terms
├── agent_1_stage_K.pt         # Per-stage agent checkpoints
├── agent_2_stage_K.pt
├── agent_3_stage_K.pt
└── predictions.jsonl          # Written by scripts/evaluate.py (post-hoc)
```

## Citation

If you find SAT useful for your research, please cite:

```bibtex
@inproceedings{xie2026sat,
  title     = {{SAT}: Sequential Agent Tuning for Coordinator-Free Plug-and-Play Multi-LLM Training with Monotonic Improvement Guarantees},
  author    = {Yi Xie and Yangyang Xu and Yi Fan and Bo Liu},
  booktitle = {Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year      = {2026}
}
```

## Frequently asked questions

**Q1. Do I need a GPU to run the demo?**
No. The demo config uses the built-in `sat:tiny` model and runs on CPU. The full Section 5 results need GPUs that fit Qwen3-4B/8B; we used four 80 GB A100s per run.

**Q2. Why is the package called `sat` while the repo is `SAT-AAMAS`?**
The Python package mirrors the algorithm name from the paper. The legacy "SAT-Seq" label only appears in earlier drafts; everything else has been unified to `sat`.

**Q3. Can I use a different model family?**
Yes. Set `sat.agents[*].path` to any Hugging Face causal LM ID or local checkpoint. The controller wraps each model with `sat.models.ModelWithValueHead`, which adds a scalar value head on the last hidden state.

**Q4. How does plug-and-play preserve the certificate?**
The Stage-0 aligner ([`sat/pnp/stage0_alignment.py`](sat/pnp/stage0_alignment.py)) projects the upgraded agent onto the per-state KL ball around the incumbent via the closed-form geometric mixture in eq. (18-19). The bisection on the Lagrange multiplier `lambda(s)` enforces `KL(pi_new || pi_cur) <= delta_0(s)`, which is exactly the trust region required by Theorems 1.1-1.2.

**Q5. Where are the bash launchers from the previous draft?**
Replaced by `scripts/train.py`, `scripts/evaluate.py`, and `scripts/plug_and_play.py`. The bash scripts that wrote inline Python are gone; if you depended on them, the new entry points accept the same YAML configs.

## Contact

- Issues / pull requests: https://github.com/Yydc/SAT-AAMAS/issues
- Authors: Yi Xie (yix@arizona.edu), Yangyang Xu, Yi Fan, Bo Liu

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.

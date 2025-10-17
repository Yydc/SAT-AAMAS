# SAT-Seq: Sequence-Aware Block-Coordinate Tuning

## 🎯 简介

SAT-Seq是一个多智能体强化学习算法，通过块坐标下降方式顺序优化多个agents，提供PAC风格的理论保证。

**核心思想**: 每次只更新一个agent，其他agents冻结，使用truncated IS重加权处理分布偏移，用quantile-based KL控制保证稳定性。

## 📁 项目结构

```
recipe/sat_seq/
├── README.md                   # 本文档
├── QUICK_START.md              # 快速开始指南
├── 答案总结.md                  # 核心问答
├── __init__.py                 # 包入口
├── real_controller.py          # 真实数据控制器（Qwen3-4B）
├── agent_scheduler.py          # Agent调度器
├── stage_coordinator.py        # 阶段协调器
├── train_dapo.sh              # DAPO训练脚本
├── run_inference.sh           # AIME24/MATH评测脚本
├── config/
│   └── sat_seq.yaml           # 配置文件
├── adv/
│   └── seqaware.py            # GAE优势估计
├── loss/
│   └── seq_ratio_loss.py      # 序列比率损失
├── kl/
│   └── quantile_kl_ctrl.py    # KL分位数控制
├── data/
│   └── reweighting.py         # IS重加权
└── monitor/
    └── certificate.py         # PAC证书

```

## 🚀 快速开始

### 1. DAPO训练

```bash
bash recipe/sat_seq/train_dapo.sh
```

### 2. AIME24/MATH评测

使用我们提供的一键评测脚本 `run_inference.sh`。

**评测AIME24数据集:**
```bash
# --dataset 参数指定评测任务
bash recipe/sat_seq/run_inference.sh --dataset aime24
```

**评测MATH数据集:**
```bash
bash recipe/sat_seq/run_inference.sh --dataset math
```

**自定义参数:**
```bash
# 指定训练好的模型目录和输出目录
bash recipe/sat_seq/run_inference.sh \
    --dataset aime24 \
    --checkpoint_dir /path/to/your/checkpoints \
    --output_dir /path/to/your/results
```

## 🔄 算法流程

### Stage级别流程

```
1. 收集on-policy数据 (π_cur)
2. 确定agent更新顺序 (scheduler)
3. 顺序更新每个agent:
   a) 激活agent_i (冻结其他)
   b) IS重加权 (处理分布偏移)
   c) 计算序列优势 (GAE + 组归一化)
   d) 优化循环:
      - 计算loss (sequence ratio + KL)
      - 梯度更新
      - 测量KL散度 (quantile-based)
      - 如果KL > target: 回溯 + 增加β
      - 否则接受更新
4. 计算PAC证书 (lower bound)
```

### Per-Agent更新

```
对于agent_i:
  1. Reweight: c_t = min(1, exp(logp_hat - logp_cur))
  2. Advantage: A_g = GAE → 组归一化 → clip(±A_clip)
  3. Loss: min{r_i*A_g, clip(r_i)*A_g} + β*KL
  4. Update: backward() + optimizer.step()
  5. KL Check: Q_{0.99}[KL] ≤ δ_target?
     - Yes: 接受
     - No: 回溯，β *= 1.5，重试
```

## ⚙️ 配置说明

### 核心参数 (`config/sat_seq.yaml`)

```yaml
algorithm:
  gamma: 0.99              # 折扣因子
  lam: 0.95               # GAE lambda
  kl_ctrl:
    kl_coef: 0.001        # 初始KL系数
    target_kl: 0.05       # 目标KL
    quantile: 0.99        # KL分位数

sat_seq:
  epsilon: 0.2            # PPO裁剪比率
  A_clip: 5.0            # 优势裁剪
  group_size: 4          # 每个prompt的响应数
  max_backtracks: 5      # 最大回溯次数
  agents:                # 3个Qwen3-4B模型
    - {name: agent_1, path: "..."}
    - {name: agent_2, path: "..."}
    - {name: agent_3, path: "..."}
```

## 📊 数据格式

### DAPO训练集 (`data/dapo/train.jsonl`)

```json
{"prompt": "问题", "chosen": "正确答案", "rejected": "错误答案"}
```

### AIME24测试集 (`data/aime24/test.jsonl`)

```json
{"problem": "问题描述", "answer": "正确答案"}
```

## 📤 输出

### 训练输出

```
outputs/sat_seq_dapo/
├── train_config.yaml
├── agent_1_stage_10.pt
├── agent_2_stage_10.pt
└── agent_3_stage_10.pt
```

### 推理输出

```
outputs/aime24_inference/
└── predictions.jsonl
    {
      "problem_id": 0,
      "predicted_answer": "28",
      "individual_answers": ["28", "28", "27"]
    }
```

### 评测输出

```
outputs/aime24_inference/
├── predictions.jsonl
└── inference_config.yaml

# predictions.jsonl 格式
{
  "problem_id": 0,
  "prompt": "Problem...",
  "pass_at_k": true,
  "avg_at_k": 0.125
}
```

## 🔧 自定义

### 修改reward函数

编辑 `real_controller.py::_compute_reward()`:

```python
def _compute_reward(self, prompt: str, response: str) -> np.ndarray:
    # Custom reward logic
    if check_answer_correct(response):
        return high_reward
    else:
        return low_reward
```

### 修改评测脚本

所有评测逻辑和参数都集中在 `recipe/sat_seq/run_inference.sh` 中，您可以直接修改该文件以满足您的需求，例如更改默认模型路径、采样温度等。

## 📝 理论背景

### PAC证书

```
Lower Bound = Info Gain - Penalties

Info Gain = Σ_i (1-γ) * E[A_i]
Penalties = Occ Shift + Estimator Bias + Finite Sample
```

### KL约束

使用quantile-based KL控制替代期望KL：
```
Q_{1-α}[KL(π_tar || π_cur)] ≤ δ_target
```

更robust，避免outliers影响。

## ⚠️ 常见问题

### Q: CUDA out of memory
```bash
# 减小batch size
TRAIN_BATCH_SIZE=256  # 在train_dapo.sh中修改
```

### Q: 模型加载失败
```bash
# 使用Hugging Face自动下载
AGENT1_PATH="Qwen/Qwen2.5-3B"  # 在train_dapo.sh中修改
```

### Q: 训练速度慢
```bash
# 减小stages数量
NUM_STAGES=2  # 在train_dapo.sh中修改
```

## 📚 更多文档

- `QUICK_START.md` - Quick start guide
- `答案总结.md` - Core Q&A (DAPO training, AIME24 inference process)
- `config/sat_seq.yaml` - Full configuration parameter explanation

## 🎓 引用

如果使用本代码，请引用：

```
@misc{sat-seq-2025,
  title={SAT-Seq: Sequence-Aware Block-Coordinate Tuning},
  year={2025}
}
```

---

**开始使用**: `bash recipe/sat_seq/train_dapo.sh` 🚀

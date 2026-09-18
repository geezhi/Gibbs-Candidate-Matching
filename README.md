<div align="center">

<h1>Gibbs Candidate Matching</h1>

<b>Candidate-based rewarded distribution matching for streaming video generation</b>

<br>

<!-- TODO: 添加 Paper / Project Page / Models 徽章 -->

</div>

## 🎯 Overview

> **TL;DR**: Gibbs Candidate Matching rolls out **multiple candidate samples** with the student model, scores every candidate with a reward model, **selects the top-1 candidate**, and applies **distribution matching distillation (DMD) on the selected candidate only**. The reward is used purely as a selection criterion — it never enters the gradient.

```
                      ┌─ candidate 1 ─┐
condition blocks ────>├─ candidate 2 ─┤──> reward model ──> argmax ──> top-1
     (no grad)        ├─ candidate 3 ─┤                                 │
                      └─ candidate 4 ─┘                                 v
                                                            distribution matching (DMD)
```

### Why this is different from advantage-weighted RL

| | Gibbs Candidate Matching (this repo) | GRPO-style |
|---|---|---|
| Use of reward | **Selection only** (pick the winner) | Normalized into advantages, **weighted into the loss** |
| Non-selected candidates | No loss term, no gradient | Contribute negative-advantage gradients |
| Gradient path | Only the top-1 candidate | All samples in the group |

Because the winner is unknown until the rewards are computed, **gradients are kept for all candidates during generation**; only the selected one actually contributes gradient in `backward()`. See [`GIBBS_CANDIDATE_MATCHING.md`](GIBBS_CANDIDATE_MATCHING.md) for the full gradient/memory analysis.

### Relation to Self Forcing / Reward Forcing

The codebase is built on top of [Self Forcing](https://github.com/guandeh17/Self-Forcing) and [Reward Forcing](https://github.com/JaydenLyh/Reward-Forcing) (autoregressive video diffusion distillation). The training objective here differs: instead of directly biasing distribution matching towards high-reward regions, Gibbs Candidate Matching first draws a set of candidates, picks the best one under the reward, and matches the distribution on that candidate.

## 📋 Table of Contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Pretrained Checkpoints](#-pretrained-checkpoints)
- [Inference](#-inference)
- [Training](#-training)
- [Method Details](#-method-details)
- [Results](#-results)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

## 🔧 Requirements

- GPU: NVIDIA GPU with at least 24GB memory for inference, 80GB memory for training.
- RAM: 64GB or more recommended.
- Linux operating system.

> Note: multi-candidate rollout holds the graphs of all candidates during the forward pass, so memory scales roughly linearly with the number of candidates. Reduce `num_rollouts` if you run out of memory.

## 🛠️ Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/geezhi/Gibbs-Candidate-Matching.git
cd Gibbs-Candidate-Matching
```

### Step 2: Create conda environment
```bash
conda create -n gibbs_cm python=3.10
conda activate gibbs_cm
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Step 4: Install the package
```bash
pip install -e .
```

## 📦 Pretrained Checkpoints

### Download Links

| Model |  Download |
|-------|----------|
| VideoReward (reward model) |  [Hugging Face](https://huggingface.co/KlingTeam/VideoReward) |
| Wan2.1-T2V-1.3B |  [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.1-T2V-14B |  [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| ODE Initialization | [Hugging Face](https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/ode_init.pt) |
| Reward Forcing (backbone init) | [Hugging Face](https://huggingface.co/JaydenLu666/Reward-Forcing-T2V-1.3B) |

### File Structure
After downloading, organize the checkpoints as follows:
```
checkpoints/
├── Videoreward/
│   ├── checkpoint-11352/
│   └── model_config.json
├── Wan2.1-T2V-1.3B/
├── Wan2.1-T2V-14B/
├── Reward-Forcing-T2V-1.3B/
└── ode_init.pt
```

### Quick Download Script
```bash
pip install "huggingface_hub[cli]"

# Download all checkpoints
bash download_checkpoints.sh
```

## 🚀 Inference
### Quick Start
```bash
# 5-seconds video inference
python inference.py \
    --num_output_frames 21 \
    --config_path configs/gibbs_candidate_matching.yaml \
    --checkpoint_path checkpoints/Reward-Forcing-T2V-1.3B/rewardforcing.pt \
    --output_folder videos/gcm-5s \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema

# 30-seconds video inference
python inference.py \
    --num_output_frames 120 \
    --config_path configs/gibbs_candidate_matching.yaml \
    --checkpoint_path checkpoints/Reward-Forcing-T2V-1.3B/rewardforcing.pt \
    --output_folder videos/gcm-30s \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```

## 🏋️ Training

### Multi-GPU Training
```bash
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=5235 --rdzv_backend=c10d  \
    --rdzv_endpoint=$MASTER_PORT train.py  --config_path configs/gibbs_candidate_matching.yaml \
    --logdir logs/gibbs_candidate_matching \
    --disable-wandb
```

### Multi-Node Training
```bash
torchrun --nnodes=$NODE_SIZE --nproc_per_node=8 --node-rank=$NODE_RANK --rdzv_id=5235 --rdzv_backend=c10d  \
    --rdzv_endpoint=$MASTER_IP:$MASTER_PORT train.py  --config_path configs/gibbs_candidate_matching.yaml \
    --logdir logs/gibbs_candidate_matching \
    --disable-wandb
```

### Configuration Files
Training configurations are in `configs/`:
- `default_config.yaml`: Default configuration
- `gibbs_candidate_matching.yaml`: Training configuration, including the candidate-matching options:

```yaml
use_multi_rollout: True   # enable multi-candidate rollout + top-1 selection
num_rollouts: 4           # number of candidate rollouts per prompt
```

Setting `use_multi_rollout: False` falls back to the single-rollout `generator_loss`.

## 🔍 Method Details

The candidate-matching logic lives in a single self-contained module:

| File | Content |
|---|---|
| `model/best_of_n.py` | `BestOfNMixin` — rollout generation, reward scoring, top-1 selection, DMD assembly |
| `model/re_dmd.py` | `ReDMD` model; provides the standard DMD loss |
| `pipeline/reward_forcing_training.py` | `inference_with_trajectory_multi_rollout` — autoregressive multi-candidate generation |
| `trainer/rewarded_distillation.py` | Training entry point |

Full documentation of the method, its difference from GRPO, and the gradient/memory behaviour: **[`GIBBS_CANDIDATE_MATCHING.md`](GIBBS_CANDIDATE_MATCHING.md)**.

## 📊 Results

<!-- TODO: 填写我们方法的结果 -->

| Method | Total Score | Quality Score | Semantic Score | Params | FPS |
|--------|----------|----------|----------|--------|-----|
| Ours | — | — | — | 1.3B | — |

## 📄 Citation

<!-- TODO: 填写论文信息 -->

```bibtex
@article{gibbs2026candidate,
  title={Gibbs Candidate Matching},
  author={TODO},
  journal={TODO},
  year={2026}
}
```

## 🙏 Acknowledgements

This project is built upon several excellent works: [CausVid](https://github.com/tianweiy/CausVid), [Self Forcing](https://github.com/guandeh17/Self-Forcing), [Reward Forcing](https://github.com/JaydenLyh/Reward-Forcing), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [VideoAlign](https://github.com/KlingTeam/VideoAlign).

We thank the authors for their great work and open-source contribution.

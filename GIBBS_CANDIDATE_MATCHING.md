# Gibbs Candidate Matching

用学生模型生成**多个候选（candidate）rollout**，用奖励模型为每个候选打分，**选出 top-1 候选**，
只对被选中的候选计算标准 distribution matching (DMD) loss。

奖励定义了候选集上的 Gibbs 分布，取 top-1 即取该分布的 mode（低温极限）。
奖励值本身**不进入梯度**，只作为筛选依据——这是 candidate matching，**不是 GRPO**。
机制上等价于 best-of-N / rejection-sampling 微调。

---

## 流程

```
                      ┌─ rollout 1 ─┐
condition blocks ────>├─ rollout 2 ─┤─> reward model 打分 ─> argmax ─> top-1
   (no grad)          ├─ rollout 3 ─┤                                   │
                      └─ rollout 4 ─┘                                   v
                                                              standard DMD loss
```

| 步骤 | 说明 |
|---|---|
| 1 | 学生模型自回归生成 `num_rollouts` 个样本，每个用新采样的噪声，共享同一条件 |
| 2 | 每个 rollout 经 VAE 解码后，用 VideoReward 的 **MQ** 维度打分 |
| 3 | 取 `argmax(rewards)`，即 reward 最高的 rollout |
| 4 | 对选中样本计算标准 DMD loss（DMD 论文 eq.7） |

---

## 与 GRPO 的区别

| | Gibbs Candidate Matching（本实现） | GRPO |
|---|---|---|
| reward 用途 | **只用于选择** winner | 归一化成 advantage，**加权进 loss** |
| 非选中 rollout | 无 loss 项，不贡献梯度 | 贡献负 advantage 的梯度 |
| 梯度来源 | 仅 top-1 一条路径 | 组内所有样本 |
| 需要 reward 的绝对值 | 不需要（只需排序） | 需要（advantage 用均值/标准差归一化） |

本仓库原先存在一条 GRPO advantage 加权分支（`compute_grpo_loss` + `grpo_weight`），
在默认配置下 `grpo_weight: 0` 即不生效；现已整体移除，只保留 Gibbs Candidate Matching 主线。

---

## 梯度与显存行为（重要）

**生成阶段：所有 rollout 都保留梯度。**

原因很直接——top-1 由奖励决定，而奖励要等所有 rollout 生成完才算得出来，
所以生成时无法预知谁是 winner，只能给全部 rollout 开梯度：

```python
# pipeline/reward_forcing_training.py
else:
    # Enable grad for all samples: we don't know which one has the highest reward
    _, denoised_pred = self.generator(...)
    break
```

分阶段看：

| 阶段 | 梯度 |
|---|---|
| condition blocks 生成 | `no_grad` |
| 中间 denoising steps | `no_grad` |
| **最后 exit step（每个 rollout）** | **保留梯度** |
| block 间 KV cache 更新 | `no_grad` |
| VAE 解码（打分用） | `no_grad` |

**反向阶段：只有 top-1 贡献梯度。**

其余 rollout 的计算图在 forward 期间仍占显存，但因为不在 loss 路径上，`backward()` 不会遍历它们。
因此**显存峰值随 `num_rollouts` 近似线性增长**——这是多候选机制的主要代价。

另外，每个 rollout 只在**一个随机选中的 exit step** 反传梯度（随机截断 BPTT），
并非整条去噪轨迹都反传。

---

## 代码结构

| 文件 | 内容 |
|---|---|
| **`model/best_of_n.py`** | **Gibbs Candidate Matching 全部逻辑**（`BestOfNMixin`）← 核心，独立可读 |
| `model/re_dmd.py` | `class ReDMD(BestOfNMixin, RewardForcingModel)`，提供标准 DMD loss |
| `model/base.py` | 模型基类（`_run_generator`、模型初始化） |
| `pipeline/reward_forcing_training.py` | `inference_with_trajectory_multi_rollout`：自回归多 rollout 生成 |
| `trainer/rewarded_distillation.py` | 训练入口，读取 `use_multi_rollout` 开关 |

调用链：

```
trainer/rewarded_distillation.py
    └─> model.generator_loss_best_of_n(...)            # model/best_of_n.py
            ├─> _run_generator_multi_rollout(...)      # model/best_of_n.py
            │       └─> inference_with_trajectory_multi_rollout(...)   # pipeline/
            ├─> _score_rollouts(...)                   # model/best_of_n.py
            └─> compute_rewarded_distribution_matching_loss(...)       # model/re_dmd.py (DMD)
```

`BestOfNMixin` 依赖宿主模型提供：`args` / `device` / `dtype` / `num_frame_per_block` /
`num_training_frames` / `inference_pipeline` / `vae` / `inferencer` /
`compute_rewarded_distribution_matching_loss`。

---

## 配置

```yaml
use_multi_rollout: True   # 开启多候选 top-1 选择；False 则走普通 generator_loss（单 rollout）
num_rollouts: 4           # 每个 prompt 的 rollout 数
```

---

## 使用

```python
# 训练步中（trainer/rewarded_distillation.py 已接好）
loss, log_dict = model.generator_loss_best_of_n(
    image_or_video_shape=shape,
    conditional_dict=conditional_dict,
    unconditional_dict=unconditional_dict,
    text_prompts=prompts,
    initial_latent=image_latent if config.i2v else None,
    num_rollouts=config.num_rollouts,
)
```

日志字段：`dmd_loss`、`rollout_reward_mean`、`rollout_reward_std`、
`rollout_reward_best`、`rollout_best_idx`。

---

## 调参与注意事项

- **`num_rollouts` 的取舍**：增大能提升 top-1 样本的质量（更接近"采样到好样本"），
  但生成开销和显存**线性增长**，且梯度只来自一条路径，收益会饱和。默认 4 是一个平衡点。
- **reward 只用 MQ 一维**（motion quality），不是 VQ+MQ+TA 的总分。
  打分前会做 `(x - MQ_mean) / MQ_std` 归一化。
- **reward 归一化与否不影响结果**：top-1 只依赖排序，单调变换不改变 argmax。
- **随机性**：rollout 数量会改变 RNG 消耗序列，因此改动 `num_rollouts` 后
  无法 bit-wise 复现原有训练轨迹（统计意义上等价）。
- 若显存吃紧，优先减小 `num_rollouts`，而不是减小 rollout 的帧数（后者会影响 DMD 的 timestep 采样范围）。

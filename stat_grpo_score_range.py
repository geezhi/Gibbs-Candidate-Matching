"""
Statistic script: for each prompt, generate num_rollouts samples using the GRPO pipeline,
compute the MQ reward score for each sample, and report the score range (max - min).
No training is performed; only the first forward pass is executed.

Usage:
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=9999 --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_PORT stat_grpo_score_range.py \
        --config_path configs/reward_forcing.yaml \
        --num_prompts 10
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from utils.distributed import fsdp_wrap, launch_distributed_job
from utils.misc import set_seed
from utils.dataset import TextDataset
from model import ReDMD


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/reward_forcing.yaml")
    parser.add_argument("--num_prompts", type=int, default=10,
                        help="Number of prompts to evaluate")
    parser.add_argument("--num_rollouts", type=int, default=None,
                        help="Override num_rollouts in config")
    parser.add_argument("--output_file", type=str, default="grpo_score_range.txt",
                        help="Path to save the statistics output")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load config
    # ------------------------------------------------------------------ #
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # Force disable wandb / saving
    config.no_save = True
    config.disable_wandb = True

    if args.num_rollouts is not None:
        config.num_rollouts = args.num_rollouts

    num_rollouts = getattr(config, "num_rollouts", 4)

    # ------------------------------------------------------------------ #
    # 2. Initialize distributed environment
    # ------------------------------------------------------------------ #
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    launch_distributed_job()
    global_rank = dist.get_rank()
    is_main_process = global_rank == 0

    dtype = torch.bfloat16 if config.mixed_precision else torch.float32
    device = torch.cuda.current_device()

    # Seed
    if config.seed == 0:
        random_seed = torch.randint(0, 10000000, (1,), device=device)
        dist.broadcast(random_seed, src=0)
        config.seed = random_seed.item()
    set_seed(config.seed + global_rank)

    # ------------------------------------------------------------------ #
    # 3. Build model (same as Trainer.__init__)
    # ------------------------------------------------------------------ #
    if is_main_process:
        print("=== Initializing ReDMD model ===")

    model = ReDMD(config, device=device)

    model.generator = fsdp_wrap(
        model.generator,
        sharding_strategy=config.sharding_strategy,
        mixed_precision=config.mixed_precision,
        wrap_strategy=config.generator_fsdp_wrap_strategy
    )
    model.real_score = fsdp_wrap(
        model.real_score,
        sharding_strategy=config.sharding_strategy,
        mixed_precision=config.mixed_precision,
        wrap_strategy=config.real_score_fsdp_wrap_strategy
    )
    model.fake_score = fsdp_wrap(
        model.fake_score,
        sharding_strategy=config.sharding_strategy,
        mixed_precision=config.mixed_precision,
        wrap_strategy=config.fake_score_fsdp_wrap_strategy
    )
    model.text_encoder = fsdp_wrap(
        model.text_encoder,
        sharding_strategy=config.sharding_strategy,
        mixed_precision=config.mixed_precision,
        wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
        cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
    )
    model.vae = model.vae.to(device=device, dtype=dtype)

    # Load generator checkpoint if specified
    if getattr(config, "generator_ckpt", False):
        if is_main_process:
            print(f"Loading pretrained generator from {config.generator_ckpt}")
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        model.generator.load_state_dict(state_dict, strict=True)

    model.eval()

    # ------------------------------------------------------------------ #
    # 4. Load prompts
    # ------------------------------------------------------------------ #
    dataset = TextDataset(config.data_path)
    num_prompts = min(args.num_prompts, len(dataset))
    prompts_list = [dataset[i]["prompts"] for i in range(num_prompts)]

    if is_main_process:
        print(f"=== Evaluating {num_prompts} prompts, {num_rollouts} samples each ===\n")

    # ------------------------------------------------------------------ #
    # 5. Per-prompt evaluation
    # ------------------------------------------------------------------ #
    all_ranges = []          # score range per prompt
    all_scores = []          # all scores per prompt (list of lists)
    all_prompts_text = []    # prompt text

    image_or_video_shape = list(config.image_or_video_shape)
    image_or_video_shape[0] = 1  # batch_size = 1

    # Cache unconditional dict
    unconditional_dict = None

    for prompt_idx, prompt in enumerate(prompts_list):
        text_prompts = [prompt]

        with torch.no_grad():
            # Encode text
            conditional_dict = model.text_encoder(text_prompts=text_prompts)

            if unconditional_dict is None:
                unconditional_dict = model.text_encoder(
                    text_prompts=[config.negative_prompt] * 1)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}

            # Generate num_rollouts samples
            rollout_latents, rollout_pixels, _, _, _ = model._run_generator_multi_rollout(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                num_rollouts=num_rollouts,
                initial_latent=None
            )

            # Compute reward for each sample
            sample_scores = []
            for i, pixels in enumerate(rollout_pixels):
                # pixels is in [-1, 1], convert to [0, 1]
                videos = (pixels + 1) / 2
                reward = model.inferencer.reward_from_frames(
                    [videos[0]],
                    text_prompts,
                    use_norm=True,
                )
                mq_score = reward['MQ'].item()
                sample_scores.append(mq_score)

        score_range = max(sample_scores) - min(sample_scores)
        all_ranges.append(score_range)
        all_scores.append(sample_scores)
        all_prompts_text.append(prompt)

        if is_main_process:
            scores_str = ", ".join(f"{s:.4f}" for s in sample_scores)
            print(f"[Prompt {prompt_idx+1:3d}/{num_prompts}] range={score_range:.4f}  "
                  f"scores=[{scores_str}]")
            print(f"  prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    # ------------------------------------------------------------------ #
    # 6. Summary statistics (main process only)
    # ------------------------------------------------------------------ #
    if is_main_process:
        import statistics
        print("\n" + "="*60)
        print("SUMMARY: Score Range (max - min) across GRPO samples")
        print("="*60)
        print(f"  Num prompts evaluated : {num_prompts}")
        print(f"  Num GRPO samples each : {num_rollouts}")
        print(f"  Mean range            : {statistics.mean(all_ranges):.4f}")
        print(f"  Median range          : {statistics.median(all_ranges):.4f}")
        print(f"  Std of range          : {statistics.stdev(all_ranges):.4f}" if len(all_ranges) > 1 else "  Std of range          : N/A")
        print(f"  Min range             : {min(all_ranges):.4f}")
        print(f"  Max range             : {max(all_ranges):.4f}")
        print("="*60)

        # Save to file
        with open(args.output_file, "w") as f:
            f.write("prompt_idx\tscore_range\tscores\tprompt\n")
            for i, (rng, scores, prompt) in enumerate(zip(all_ranges, all_scores, all_prompts_text)):
                scores_str = ",".join(f"{s:.6f}" for s in scores)
                f.write(f"{i+1}\t{rng:.6f}\t{scores_str}\t{prompt}\n")
            f.write("\n# Summary\n")
            f.write(f"mean_range\t{statistics.mean(all_ranges):.6f}\n")
            f.write(f"median_range\t{statistics.median(all_ranges):.6f}\n")
            f.write(f"min_range\t{min(all_ranges):.6f}\n")
            f.write(f"max_range\t{max(all_ranges):.6f}\n")
        print(f"\nResults saved to: {args.output_file}")

        # ------------------------------------------------------------------ #
        # 7. Plot distribution of score ranges
        # ------------------------------------------------------------------ #
        plot_path = args.output_file.replace(".txt", "_distribution.png")
        _plot_range_distribution(all_ranges, all_scores, num_rollouts, plot_path)


def _plot_range_distribution(all_ranges, all_scores, num_rollouts, save_path):
    """
    Draw two subplots:
      Left : histogram + KDE of score ranges across all prompts
      Right: box plot of raw scores per sample index across all prompts
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib / numpy not available, skipping plot.")
        return

    ranges = np.array(all_ranges)
    # all_scores: list of lists, shape [num_prompts, num_rollouts]
    scores_arr = np.array(all_scores)  # [num_prompts, num_rollouts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"GRPO Score Range Distribution\n"
        f"({len(all_ranges)} prompts × {num_rollouts} samples each)",
        fontsize=13
    )

    # ---- Left: histogram of ranges ----
    ax = axes[0]
    n_bins = max(10, len(ranges) // 5)
    ax.hist(ranges, bins=n_bins, color="steelblue", edgecolor="white", alpha=0.85, label="count")

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(ranges.min() - 0.01, ranges.max() + 0.01, 300)
        kde_y = gaussian_kde(ranges)(kde_x)
        # scale KDE to histogram height
        bin_width = (ranges.max() - ranges.min()) / n_bins
        ax2 = ax.twinx()
        ax2.plot(kde_x, kde_y, color="tomato", linewidth=2, label="KDE")
        ax2.set_ylabel("Density", color="tomato")
        ax2.tick_params(axis="y", labelcolor="tomato")
        ax2.set_ylim(bottom=0)
    except ImportError:
        pass  # scipy not available, skip KDE

    ax.axvline(ranges.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"mean={ranges.mean():.3f}")
    ax.axvline(np.median(ranges), color="green", linestyle=":", linewidth=1.5,
               label=f"median={np.median(ranges):.3f}")
    ax.set_xlabel("Score Range (max − min)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Score Ranges", fontsize=11)
    ax.legend(fontsize=9)

    # ---- Right: box plot of scores per sample index ----
    ax = axes[1]
    data_per_sample = [scores_arr[:, i] for i in range(num_rollouts)]
    bp = ax.boxplot(data_per_sample, patch_artist=True, notch=False,
                    medianprops=dict(color="red", linewidth=2))
    colors = plt.cm.tab10.colors
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Sample Index (within GRPO group)", fontsize=11)
    ax.set_ylabel("MQ Score", fontsize=11)
    ax.set_title("Score Distribution per Sample Slot", fontsize=11)
    ax.set_xticks(range(1, num_rollouts + 1))
    ax.set_xticklabels([f"Sample {i+1}" for i in range(num_rollouts)])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribution plot saved to: {save_path}")


if __name__ == "__main__":
    main()

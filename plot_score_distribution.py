"""
Standalone plotting script: read the output txt from stat_grpo_score_range.py
and draw two distribution plots side by side:
  Left : histogram + KDE of score ranges (max - min) across all prompts
  Right: histogram + KDE of score means across all prompts

Usage:
    python plot_score_distribution.py --input grpo_score_range.txt
    python plot_score_distribution.py --input grpo_score_range.txt --output my_plot.png
"""

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="grpo_score_range.txt",
                        help="Path to the txt file produced by stat_grpo_score_range.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path. Defaults to <input_stem>_distribution.png")
    return parser.parse_args()


def load_txt(path):
    """Parse the tab-separated txt file, return (ranges, means, all_scores)."""
    all_ranges = []
    all_means = []
    all_scores = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip header, empty lines, and summary lines starting with '#'
            if not line or line.startswith("#") or line.startswith("prompt_idx"):
                continue
            # Skip summary key=value lines (no tab-separated 4 columns)
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                # parts: [prompt_idx, score_range, scores, prompt]
                score_range = float(parts[1])
                scores = [float(s) for s in parts[2].split(",")]
            except ValueError:
                continue

            all_ranges.append(score_range)
            all_means.append(np.mean(scores))
            all_scores.append(scores)

    return np.array(all_ranges), np.array(all_means), all_scores


def plot(ranges, means, all_scores, save_path, input_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "serif"]
    plt.rcParams["mathtext.fontset"] = "stix"  # math symbols use Times-style

    num_prompts = len(ranges)
    num_samples = len(all_scores[0]) if all_scores else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # fig.suptitle(
    #     f"Multi-rollout Score Distribution",
    #     fontsize=18
    # )

    def draw_hist_kde(ax, data, color_hist, color_kde, xlabel, title):
        n_bins = max(10, len(data) // 5)
        # density=True: y-axis is probability density, consistent with KDE
        ax.hist(data, bins=n_bins, density=True, color=color_hist, edgecolor="white",
                alpha=0.80, label="histogram")

        # KDE overlay on the same axis
        try:
            from scipy.stats import gaussian_kde
            x_min, x_max = data.min(), data.max()
            margin = (x_max - x_min) * 0.05 + 1e-6
            kde_x = np.linspace(x_min - margin, x_max + margin, 400)
            kde_y = gaussian_kde(data)(kde_x)
            ax.plot(kde_x, kde_y, color=color_kde, linewidth=2, label="KDE")
        except ImportError:
            pass  # scipy not available

        ax.axvline(data.mean(), color="orange", linestyle="--", linewidth=1.8,
                   label=f"mean = {data.mean():.3f}")
        ax.axvline(np.median(data), color="green", linestyle=":", linewidth=1.8,
                   label=f"median = {np.median(data):.3f}")
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("Density", fontsize=15)
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(fontsize=13, loc="upper right")

        # Annotate std
        ax.text(0.02, 0.97, f"std = {data.std():.3f}\nmin = {data.min():.3f}\nmax = {data.max():.3f}",
                transform=ax.transAxes, fontsize=13, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # ---- Left: score range distribution ----
    draw_hist_kde(
        axes[0], ranges,
        color_hist="steelblue", color_kde="tomato",
        xlabel="Score Range (max − min)",
        title="Distribution of Score Ranges"
    )

    # ---- Right: score mean distribution ----
    draw_hist_kde(
        axes[1], means,
        color_hist="mediumpurple", color_kde="darkorange",
        xlabel="Score Mean",
        title="Distribution of Score Means"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {save_path}")


def main():
    args = parse_args()

    if args.output is None:
        stem = args.input.rsplit(".", 1)[0]
        args.output = stem + "_distribution.pdf"

    print(f"Loading data from: {args.input}")
    ranges, means, all_scores = load_txt(args.input)
    print(f"  Loaded {len(ranges)} prompts")
    print(f"  Range  — mean={ranges.mean():.4f}, std={ranges.std():.4f}, "
          f"min={ranges.min():.4f}, max={ranges.max():.4f}")
    print(f"  Mean   — mean={means.mean():.4f}, std={means.std():.4f}, "
          f"min={means.min():.4f}, max={means.max():.4f}")

    plot(ranges, means, all_scores, args.output, args.input)


if __name__ == "__main__":
    main()

"""
Comparison inference script.

Workflow:
  1. Use self-forcing to generate a 5s prefix (81 frames @ 16fps).
  2. Use the same prefix latent as initial_latent for three models:
       - self-forcing  (--self_forcing_checkpoint)
       - reward-forcing (--reward_forcing_checkpoint)
       - ours           (--ours_checkpoint)
  3. Each model generates the next 81 frames conditioned on the prefix KV-cache.
  4. Save three videos: prefix+self_forcing, prefix+reward_forcing, prefix+ours.

Usage:
  python inference_compare.py \
      --self_forcing_checkpoint /path/to/self_forcing.pt \
      --reward_forcing_checkpoint /path/to/rewardforcing.pt \
      --ours_checkpoint /path/to/ours.pt \
      --prompt "A cat playing on the grass" \
      --output_folder videos/compare \
      --seed 42
"""

import argparse
import os
import torch
from torchvision.io import write_video
from einops import rearrange
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from utils.misc import set_seed

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--self_forcing_checkpoint", type=str, required=True,
                    help="Path to self-forcing model checkpoint (.pt), used for prefix generation and comparison")
parser.add_argument("--reward_forcing_checkpoint", type=str, required=True,
                    help="Path to reward-forcing checkpoint (.pt)")
parser.add_argument("--ours_checkpoint", type=str, required=True,
                    help="Path to 'ours' model checkpoint (.pt)")
parser.add_argument("--config_path", type=str, default="configs/reward_forcing_copy.yaml",
                    help="Config file (used for denoising_step_list, model_kwargs, etc.)")
parser.add_argument("--prompt", type=str, required=True,
                    help="Text prompt for generation")
parser.add_argument("--output_folder", type=str, default="videos/compare1",
                    help="Output folder for the three comparison videos")
parser.add_argument("--prefix_frames", type=int, default=21,
                    help="Number of frames for the prefix clip (default 81 = 5s @ 16fps)")
parser.add_argument("--suffix_frames", type=int, default=81,
                    help="Number of frames each model generates after the prefix")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--use_ema", action="store_true",
                    help="Load EMA weights from checkpoints that contain 'generator_ema' key")
args = parser.parse_args()

set_seed(args.seed)
device = torch.device("cuda")
torch.set_grad_enabled(False)

gpu = device
low_memory = get_cuda_free_memory_gb(gpu) < 40
print(f"Free VRAM: {get_cuda_free_memory_gb(gpu):.1f} GB  |  low_memory={low_memory}")

os.makedirs(args.output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.load(args.config_path)
config = OmegaConf.merge(default_config, config)

# ---------------------------------------------------------------------------
# Helper: load a checkpoint into a pipeline's generator
# ---------------------------------------------------------------------------
def load_checkpoint(pipeline: CausalInferencePipeline, ckpt_path: str, use_ema: bool = False):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"  [skip] checkpoint not found: {ckpt_path}")
        return
    print(f"  Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    def strip_fsdp(sd):
        new_sd = {}
        for k, v in sd.items():
            new_k = k.replace("model._fsdp_wrapped_module.", "model.")
            new_sd[new_k] = v
        return new_sd

    if "generator_ema" in state_dict and (use_ema or "generator" not in state_dict):
        print("  Using 'generator_ema' weights.")
        sd = strip_fsdp(state_dict["generator_ema"])
        pipeline.generator.load_state_dict(sd, strict=False)
    elif "generator" in state_dict:
        sd = strip_fsdp(state_dict["generator"])
        pipeline.generator.load_state_dict(sd)
    else:
        sd = strip_fsdp(state_dict)
        pipeline.generator.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Helper: build a CausalInferencePipeline (shares text_encoder & vae)
# ---------------------------------------------------------------------------
def build_pipeline(config, device, shared_text_encoder=None, shared_vae=None):
    pipeline = CausalInferencePipeline(
        config,
        device=device,
        text_encoder=shared_text_encoder,
        vae=shared_vae,
    )
    return pipeline


# ---------------------------------------------------------------------------
# Step 1: Build self-forcing pipeline and generate the 5s prefix
# ---------------------------------------------------------------------------
print("\n=== Step 1: Building self-forcing pipeline ===")
sf_pipeline = build_pipeline(config, device)
load_checkpoint(sf_pipeline, args.self_forcing_checkpoint, use_ema=args.use_ema)
sf_pipeline = sf_pipeline.to(dtype=torch.bfloat16)

if low_memory:
    DynamicSwapInstaller.install_model(sf_pipeline.text_encoder, device=gpu)
else:
    sf_pipeline.text_encoder.to(device=gpu)
sf_pipeline.generator.to(device=gpu)
sf_pipeline.vae.to(device=gpu)

print(f"Generating {args.prefix_frames}-frame prefix with self-forcing ...")
prefix_noise = torch.randn(
    [1, args.prefix_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
)
prefix_video_pixel, prefix_latent = sf_pipeline.inference(
    noise=prefix_noise,
    text_prompts=[args.prompt],
    return_latents=True,
    low_memory=low_memory,
)
# prefix_latent: [1, prefix_frames, 16, 60, 104]
# prefix_video_pixel: [1, prefix_frames, 3, H, W], range [0,1]
sf_pipeline.vae.model.clear_cache()
print(f"  prefix_latent shape: {prefix_latent.shape}")

# Save prefix video for reference
prefix_video_out = (255.0 * prefix_video_pixel).byte()
prefix_video_out = rearrange(prefix_video_out, 'b t c h w -> b t h w c')
write_video(
    os.path.join(args.output_folder, "prefix_self_forcing.mp4"),
    prefix_video_out[0].cpu(), fps=16
)
print("  Saved prefix video.")

# Keep shared text_encoder and vae to avoid reloading
shared_text_encoder = sf_pipeline.text_encoder
shared_vae = sf_pipeline.vae

# ---------------------------------------------------------------------------
# Step 2: Define the three models and their checkpoints
# ---------------------------------------------------------------------------
model_configs = [
    {
        "name": "self_forcing",
        "ckpt": args.self_forcing_checkpoint,
    },
    {
        "name": "reward_forcing",
        "ckpt": args.reward_forcing_checkpoint,
    },
    {
        "name": "ours",
        "ckpt": args.ours_checkpoint,
    },
]

# ---------------------------------------------------------------------------
# Step 3: For each model, run inference conditioned on the prefix latent
# ---------------------------------------------------------------------------
for model_cfg in model_configs:
    model_name = model_cfg["name"]
    ckpt_path = model_cfg["ckpt"]
    print(f"\n=== Generating suffix with model: {model_name} ===")

    # Build a fresh pipeline (reuse text_encoder and vae)
    pipeline = build_pipeline(config, device,
                               shared_text_encoder=shared_text_encoder,
                               shared_vae=shared_vae)
    load_checkpoint(pipeline, ckpt_path, use_ema=args.use_ema)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=gpu)

    # Suffix noise: [1, suffix_frames, 16, 60, 104]
    suffix_noise = torch.randn(
        [1, args.suffix_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    # Use prefix_latent as initial_latent so the KV cache is pre-filled with prefix context
    suffix_video_pixel, suffix_latent = pipeline.inference(
        noise=suffix_noise,
        text_prompts=[args.prompt],
        initial_latent=prefix_latent,
        return_latents=True,
        low_memory=low_memory,
    )
    pipeline.vae.model.clear_cache()

    # Concatenate prefix + suffix pixels
    # NOTE: when initial_latent is provided, inference() returns a video that includes
    # the decoded initial_latent frames followed by the newly generated frames.
    # So we slice off the first prefix_pixel_frames to avoid duplication.
    # prefix_frames is in latent space; convert to pixel space: (prefix_frames - 1) * 4 + 1
    prefix_pixel_frames = (args.prefix_frames - 1) * 4 + 1
    suffix_only = suffix_video_pixel[:, prefix_pixel_frames:, ...]  # [1, suffix_frames_pixel, C, H, W]
    full_video = torch.cat([prefix_video_pixel, suffix_only], dim=1)  # [1, T_total, C, H, W]
    full_video_out = (255.0 * full_video).byte()
    full_video_out = rearrange(full_video_out, 'b t c h w -> b t h w c')

    out_path = os.path.join(args.output_folder, f"{model_name}.mp4")
    write_video(out_path, full_video_out[0].cpu(), fps=16)
    print(f"  Saved: {out_path}  (total frames: {full_video.shape[1]})")

    # Free generator memory before loading next model
    del pipeline
    torch.cuda.empty_cache()

print("\n=== Done! All three comparison videos saved to:", args.output_folder, "===")

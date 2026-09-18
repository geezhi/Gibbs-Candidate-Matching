"""
Best-of-N training (multi-rollout + top-1 selection) for Reward Forcing.

Method
------
1. Roll out `num_rollouts` videos with the *student* model, each starting from freshly
   sampled noise while sharing the same condition.
2. Score every rollout with a reward model (VideoReward, MQ dimension).
3. Take the highest-reward rollout (top-1) and apply the standard DMD loss on it.

The reward value never enters the gradient: it is used purely as a selection criterion.
This is best-of-N / rejection-sampling fine-tuning, **not** GRPO -- there is no advantage
weighting and no loss term on the non-selected rollouts.

Why every rollout keeps its graph
---------------------------------
The winner is unknown until the rewards are computed, and the rewards only exist after all
rollouts have been generated. Gradients must therefore stay enabled during generation for
*all* rollouts (at the exit step); see `inference_with_trajectory_multi_rollout`. During
`backward()` only the top-1 rollout actually contributes gradient -- the graphs of the
other rollouts are retained during the forward pass but are not traversed.

Usage
-----
Mix into a model class that already implements `compute_rewarded_distribution_matching_loss`
(the standard DMD loss):

    class ReDMD(BestOfNMixin, RewardForcingModel):
        ...

and call `generator_loss_best_of_n(...)` from the training step.
"""

from typing import Tuple
import torch
import torch.distributed as dist


class BestOfNMixin:
    """
    Multi-rollout best-of-N training logic.

    Expects the host model to provide:
        - self.args, self.device, self.dtype
        - self.num_frame_per_block, self.num_training_frames
        - self.inference_pipeline and self._initialize_inference_pipeline()
        - self.vae
        - self.inferencer: reward-model wrapper exposing `reward_from_frames`
        - self.compute_rewarded_distribution_matching_loss(...): the standard DMD loss
    """

    def _run_generator_multi_rollout(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        num_rollouts: int = 4,
        initial_latent=None
    ):
        """
        Multi-rollout variant of `_run_generator`.

        Generates the condition blocks (no grad), then rolls out `num_rollouts` different
        last-21-frame segments: each rollout starts from freshly sampled noise and shares
        the same condition (the KV cache is restored before every rollout).

        Gradients are enabled at the exit step for *all* rollouts rather than only for the
        one that will be selected. This is required by best-of-N selection: the winner is
        determined by the rewards, and the rewards only exist after the rollouts are
        generated, so the winner cannot be known during generation.

        Returns:
            rollout_latents: list of `num_rollouts` tensors, each [B, 21, C, H, W]
            rollout_pixels:  list of `num_rollouts` decoded pixel tensors (no grad)
            gradient_mask:   optional bool mask over the generated frames
            denoised_timestep_from, denoised_timestep_to: timestep range for the DMD loss
        """
        assert getattr(self.args, "backward_simulation", True)
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        noise_shape[1] = num_generated_frames

        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        (
            condition_output,
            rollout_outputs,
            denoised_timestep_from,
            denoised_timestep_to,
            exit_step,
            condition_start_frame_abs,
        ) = self.inference_pipeline.inference_with_trajectory_multi_rollout(
            noise=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
            num_rollouts=num_rollouts,
            **conditional_dict,
        )

        if num_generated_frames != min_num_frames:
            gradient_mask = torch.ones(
                [image_or_video_shape[0], 21, *image_or_video_shape[2:]], dtype=torch.bool, device=self.device)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        rollout_latents = []
        rollout_pixels = []
        for latent_21 in rollout_outputs:
            latent_21 = latent_21.to(self.dtype)
            rollout_latents.append(latent_21)
            with torch.no_grad():
                pixels = self.vae.decode_to_pixel(latent_21).to(self.dtype)
            rollout_pixels.append(pixels)

        return rollout_latents, rollout_pixels, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _score_rollouts(self, rollout_pixels: list, text_prompts: list) -> torch.Tensor:
        """
        Score every rollout with the reward model.

        Input:
            - rollout_pixels: list of `num_rollouts` pixel tensors, each in [-1, 1].
            - text_prompts: list of text prompts.
        Output:
            - rewards: tensor of shape [num_rollouts].
        """
        rewards = []
        with torch.no_grad():
            for pixels in rollout_pixels:
                # decode_to_pixel returns [-1, 1]; the reward processor expects [0, 1]
                videos = (pixels + 1) / 2
                reward = self.inferencer.reward_from_frames(
                    [videos[0]],
                    [text_prompts[0]],
                    use_norm=True,
                )
                rewards.append(reward['MQ'])
        return torch.stack(rewards)

    def generator_loss_best_of_n(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        text_prompts: list,
        clean_latent: torch.Tensor = None,
        initial_latent: torch.Tensor = None,
        num_rollouts: int = 4,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Multi-rollout best-of-N training step.

        1. Roll out `num_rollouts` samples with the student model (all keep their graphs at
           the exit step, since the winner is unknown before scoring).
        2. Score each rollout with the reward model.
        3. Take the highest-reward rollout (top-1) and apply the standard DMD loss on it.

        The reward value itself never enters the gradient: it is used purely as a selection
        criterion. Only the top-1 rollout contributes gradient to the generator, while the
        other rollouts still hold their graphs in memory during the forward pass.

        `clean_latent` is unused and kept only for signature compatibility with
        `generator_loss`; backward simulation removes the need for real data.
        """
        # Step 1: Roll out several samples with the student model
        rollout_latents, rollout_pixels, gradient_mask, ts_from, ts_to = \
            self._run_generator_multi_rollout(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                num_rollouts=num_rollouts,
                initial_latent=initial_latent
            )

        # Step 2: Score every rollout (no grad)
        rewards = self._score_rollouts(rollout_pixels, text_prompts)  # [num_rollouts]

        # Step 3: Best-of-N selection
        best_idx = int(rewards.argmax().item())
        best_latent = rollout_latents[best_idx]

        # Step 4: Standard DMD loss on the selected rollout
        dmd_loss, log_dict = self.compute_rewarded_distribution_matching_loss(
            image_or_video=best_latent,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=ts_from,
            denoised_timestep_to=ts_to,
        )

        log_dict.update({
            "dmd_loss": dmd_loss.detach(),
            "rollout_reward_mean": rewards.mean().detach(),
            "rollout_reward_std": rewards.std().detach(),
            "rollout_reward_best": rewards[best_idx].detach(),
            "rollout_best_idx": torch.tensor(best_idx, dtype=torch.float32),
        })

        return dmd_loss, log_dict

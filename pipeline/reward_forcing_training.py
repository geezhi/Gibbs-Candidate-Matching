# same as self forcing training pipeline
from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class RewardForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # if self.kv_cache1 is None:
        #     self._initialize_kv_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device,
        #     )
        #     self._initialize_crossattn_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device
        #     )
        # else:
        #     # reset cross attn cache
        #     for block_index in range(self.num_transformer_blocks):
        #         self.crossattn_cache[block_index]["is_init"] = False
        #     # reset kv cache
        #     for block_index in range(len(self.kv_cache1)):
        #         self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #         self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _clone_kv_cache(self):
        """Deep-clone the current KV cache state for later restoration."""
        cloned = []
        for entry in self.kv_cache1:
            cloned.append({
                "k": entry["k"].clone(),
                "v": entry["v"].clone(),
                "global_end_index": entry["global_end_index"].clone(),
                "local_end_index": entry["local_end_index"].clone(),
            })
        return cloned

    def _clone_crossattn_cache(self):
        """Deep-clone the current cross-attention cache state for later restoration."""
        cloned = []
        for entry in self.crossattn_cache:
            cloned.append({
                "k": entry["k"].clone(),
                "v": entry["v"].clone(),
                "is_init": entry["is_init"],
            })
        return cloned

    def _restore_kv_cache(self, saved_cache):
        """Restore KV cache from a previously saved clone."""
        for i, entry in enumerate(saved_cache):
            self.kv_cache1[i]["k"].copy_(entry["k"])
            self.kv_cache1[i]["v"].copy_(entry["v"])
            self.kv_cache1[i]["global_end_index"].copy_(entry["global_end_index"])
            self.kv_cache1[i]["local_end_index"].copy_(entry["local_end_index"])

    def _restore_crossattn_cache(self, saved_cache):
        """Restore cross-attention cache from a previously saved clone."""
        for i, entry in enumerate(saved_cache):
            self.crossattn_cache[i]["k"].copy_(entry["k"])
            self.crossattn_cache[i]["v"].copy_(entry["v"])
            self.crossattn_cache[i]["is_init"] = entry["is_init"]

    def inference_with_trajectory_multi_rollout(
            self,
            noise,
            num_rollouts: int = 4,
            initial_latent=None,
            **conditional_dict
    ):
        """
        Multi-rollout variant: generate the condition blocks (no grad), then roll out
        `num_rollouts` different last-21-frame segments, each starting from freshly sampled
        noise while sharing the same condition (the KV cache is restored before each rollout).

        Every rollout keeps its graph alive at the exit step. This is deliberate: the
        best-of-N winner is decided by the rewards, and the rewards only exist after all
        rollouts have been generated, so the winner cannot be known in advance.

        Returns:
            condition_output: condition frames (no grad)
            rollout_outputs: list of num_rollouts tensors, each [B, 21, C, H, W]
            denoised_timestep_from, denoised_timestep_to: timestep range for DMD loss
            exit_step: the exit denoising step index
            condition_start_frame_abs: frame index where the last 21 frames start
        """
        import torch
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        # Step 1: Initialize KV cache
        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        # Step 2: Cache initial latent (i2v)
        current_start_frame = 0
        condition_output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype
        )
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            condition_output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Determine block layout and exit flags
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)

        # The last 21 frames correspond to the last (21 // num_frame_per_block) blocks
        last_21_blocks = 21 // self.num_frame_per_block
        condition_blocks = len(all_num_frames) - last_21_blocks
        condition_start_frame_abs = num_input_frames + sum(all_num_frames[:condition_blocks])

        # Step 4: Generate condition blocks (no grad, update KV cache)
        for block_index in range(condition_blocks):
            current_num_frames = all_num_frames[block_index]
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])
                timestep = torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep

                with torch.no_grad():
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    if not exit_flag:
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        break

            condition_output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Update KV cache with denoised output
            context_timestep = torch.ones_like(timestep) * self.context_noise
            denoised_pred_ctx = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred_ctx,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += current_num_frames

        # Step 5: Save KV cache state after condition blocks
        saved_kv_cache = self._clone_kv_cache()
        saved_crossattn_cache = self._clone_crossattn_cache()
        saved_start_frame = current_start_frame

        # Compute denoised timestep info
        last_block_exit = exit_flags[condition_blocks] if not self.same_step_across_blocks else exit_flags[0]
        if last_block_exit == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[last_block_exit].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[last_block_exit + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[last_block_exit].cuda()).abs(), dim=0).item()

        # Step 6: Generate num_rollouts different last-21-frame segments
        rollout_outputs = []
        for sample_idx in range(num_rollouts):
            # Restore KV cache to condition state
            self._restore_kv_cache(saved_kv_cache)
            self._restore_crossattn_cache(saved_crossattn_cache)
            current_start_frame = saved_start_frame

            # Sample new noise for the last 21 frames
            sample_noise = torch.randn(
                [batch_size, 21, num_channels, height, width], device=noise.device, dtype=noise.dtype)

            sample_output = torch.zeros(
                [batch_size, 21, num_channels, height, width],
                device=noise.device, dtype=noise.dtype
            )
            local_start = 0

            for block_index in range(condition_blocks, len(all_num_frames)):
                current_num_frames = all_num_frames[block_index]
                noisy_input = sample_noise[:, local_start:local_start + current_num_frames]

                for index, current_timestep in enumerate(self.denoising_step_list):
                    if self.same_step_across_blocks:
                        exit_flag = (index == exit_flags[0])
                    else:
                        exit_flag = (index == exit_flags[block_index])
                    timestep = torch.ones(
                        [batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep

                    if not exit_flag:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                            next_timestep = self.denoising_step_list[index + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # Enable grad for all samples: we don't know which one has the highest reward
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        break

                sample_output[:, local_start:local_start + current_num_frames] = denoised_pred

                # Update KV cache for subsequent blocks within this sample
                if block_index < len(all_num_frames) - 1:
                    context_timestep = torch.ones_like(timestep) * self.context_noise
                    denoised_pred_ctx = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        context_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                    with torch.no_grad():
                        self.generator(
                            noisy_image_or_video=denoised_pred_ctx,
                            conditional_dict=conditional_dict,
                            timestep=context_timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )

                current_start_frame += current_num_frames
                local_start += current_num_frames

            rollout_outputs.append(sample_output)

        return (
            condition_output[:, :condition_start_frame_abs],
            rollout_outputs,
            denoised_timestep_from,
            denoised_timestep_to,
            last_block_exit + 1,
            condition_start_frame_abs,
        )

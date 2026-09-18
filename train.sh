torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=5235 --rdzv_backend=c10d  \
    --rdzv_endpoint=$MASTER_PORT train.py  --config_path configs/gibbs_candidate_matching.yaml \
    --logdir logs/reward_forcing \
    --disable-wandb


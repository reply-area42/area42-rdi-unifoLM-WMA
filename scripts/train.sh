# NCCL configuration
# export NCCL_DEBUG=debug
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_NET_GDR_LEVEL=3
# export CUDA_LAUNCH_BLOCKING=1

# export NCCL_TOPO_FILE=/tmp/topo.txt
# export MASTER_ADDR="master.ip."
# export MASTER_PROT=12366


# args
name="sprite_wma_v1"
config_file=configs/train/config.yaml
save_root="outputs/"

mkdir -p $save_root/$name

CUDA_VISIBLE_DEVICES=0 python3 ./scripts/trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices 1 \
--total_gpus=1 \
lightning.trainer.num_nodes=1

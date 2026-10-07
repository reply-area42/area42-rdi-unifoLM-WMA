model_name=testing
ckpt=/home/humanoid42/Workspace/area42-rdi-unifoLM-WMA/outputs/sprite_wma_v1/checkpoints/trainstep_checkpoints/epoch=1521-step=35000.ckpt
config=configs/inference/world_model_decision_making.yaml
seed=123
res_dir="/home/humanoid42/Workspace/area42-rdi-unifoLM-WMA/outputs/sprite_wma_v1_eval"
datasets=(
    "sprite_wma_v1"
)


for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/real_eval_server.py \
    --seed ${seed} \
    --ckpt_path $ckpt \
    --config $config \
    --savedir "${res_dir}/${dataset}/${model_name}/videos" \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 1.0 \
    --ddim_steps 16 \
    --ddim_eta 1.0 \
    --video_length 16 \
    --frame_stride 2 \
    --timestep_spacing 'uniform_trailing' \
    --guidance_rescale 0.7 \
    --perframe_ae
done

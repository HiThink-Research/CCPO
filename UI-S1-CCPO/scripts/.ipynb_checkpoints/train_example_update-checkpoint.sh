
# rlog
# source /opt/conda/etc/profile.d/conda.sh

export NCCL_DEBUG=INFO          # 输出详细NCCL日志定位问题
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理机制
export PYTHONFAULTHANDLER=1     # 开启Python层错误堆
export HYDRA_FULL_ERROR=1
# conda activate ui-s1

export PYTHONPATH=.

set -x

export VLLM_USE_V1=1

ENGINE=${1:-vllm}
GPUS=${GPUS:-8}              # 使用的GPU数量 (默认8)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES
mode="mean_std_norm"
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"



GAMMA=0.5
DAPO=True
DAPO_THRESHOLD=0.3
PATCH_THRESHOLD=2
EXPERIMENT_NAME="qwenvl_uis1_DAPO_${DAPO}_${DAPO_THRESHOLD}_patch_${PATCH_THRESHOLD}_gamma_${GAMMA}"
if [ $MASTER_ADDR ];then
    echo $MASTER_ADDR
    echo $MASTER_PORT
    echo $WORLD_SIZE
    echo $RANKa
else
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
    export WORLD_SIZE=1  # 多机时再改，这里只调节点内GPU数
    export RANK=0
fi
BATCH_SIZE=$((WORLD_SIZE * GPUS))

set -x
ray stop


if [ "$RANK" == "0" ]; then
    # 启动Ray头节点
    ray start --head --node-ip-address=$MASTER_ADDR
    python3 -m verl.trainer.main_dapo \
        --config-path="$CONFIG_PATH" \
        --config-name='traj_grpo' \
        algorithm.adv_estimator=uis1 \
        data.train_files=/mnt/HithinkOmniSSD/user_workspace/yinjiong/dataset/GUIOdyssey/multi_turn_anno/train_goal_steps_debug.jsonl \
        data.val_files=/mnt/HithinkOmniSSD/user_workspace/yinjiong/dataset/GUIOdyssey/multi_turn_anno/test_goal_steps_debug.jsonl \
        data.train_batch_size=${BATCH_SIZE} \
        data.val_batch_size=$((8 * BATCH_SIZE)) \
        data.max_prompt_length=12288 \
        data.max_response_length=512 \
        data.truncation='error' \
        actor_rollout_ref.model.path=/mnt/HithinkOmniSSD/user_workspace/yinjiong/code/GUI-Agent-Compression/SimpAgent/checkpoint/GUI-O-lora-3B-09-10/4AO_bz8_sequence_baseline_lora/merge/checkpoint-63805\
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
        actor_rollout_ref.actor.use_fixed_num_mini_batches=true \
        actor_rollout_ref.actor.fixed_num_mini_batches=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.max_model_len=32678  \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.limit_images=2 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        algorithm.gamma=$GAMMA \
        algorithm.uis1.step_advantage_w=1.0 \
        algorithm.uis1.mode=$mode \
        algorithm.patch_threshold=$PATCH_THRESHOLD \
        algorithm.filter_groups.enable=$DAPO \
        algorithm.filter_groups.metric='seq_future_reward' \
        algorithm.filter_groups.std_threshold=$DAPO_THRESHOLD \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='gui_traj_grpo' \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=${GPUS} \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=5 \
        trainer.test_freq=10 \
        trainer.val_before_train=False \
        trainer.total_epochs=3 $@
    ray stop
else
    # 连接到Ray头节点
    ray start --address="$MASTER_ADDR:6379"
    # 持续检测集群状态
    while ray status >/dev/null 2>&1; do
        sleep 10
    done
    echo "Ray cluster stopped, exiting worker script"
    exit 0
fi

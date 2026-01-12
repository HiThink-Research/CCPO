# -*- coding: utf-8 -*-
set -x
nvidia-smi

# 1) light env
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export HYDRA_FULL_ERROR=0

# export WANDB_API_KEY="xxx"

# pip install -e .
# pip install vllm==0.9.2
# pip install transformers==4.52.3
# pip install flash-attn==2.7.4.post1 --no-build-isolation

# source /opt/conda/etc/profile.d/conda.sh
export PYTHONPATH=.
export VLLM_USE_V1=1

# export WANDB_PROJECT="CCPO-AITW"
#export WANDB_MODE="online"  # or "offline" for offline logging

# Set GLOO_SOCKET_IFNAME from NCCL_SOCKET_IFNAME if not already set
if [ -n "$NCCL_SOCKET_IFNAME" ] && [ -z "$GLOO_SOCKET_IFNAME" ]; then
    export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
fi

ENGINE=${1:-vllm}
mode="mean_std_norm"
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"

GAMMA=0.3
DAPO=True
DAPO_THRESHOLD=0.1
PATCH_THRESHOLD=1
ROLLOUT=8
NODE=2
LR=5e-7
IMAGE=2
KL_LOSS=0.0001
ENTROPY=0.0
CHECKPOINT="7B_1AO_lora_final_coord_local"
CHECKPOINT_STEP=500
REWARD="CCPO_MAX_CR"


EXPERIMENT_NAME="${CHECKPOINT}_${CHECKPOINT_STEP}_${DAPO_THRESHOLD}_patch_${PATCH_THRESHOLD}_gamma_${GAMMA}_LR_${LR}_N_${ROLLOUT}_KL_${KL_LOSS}_${ENTROPY}_${NODE}node_$((IMAGE-1))AO_${REWARD}"

if [ $MASTER_ADDR ];then
    echo $MASTER_ADDR
    echo $MASTER_PORT
    echo $WORLD_SIZE
    echo $RANK

    echo "检测到Master已设置的分布式环境变量:"
    echo "  MASTER_ADDR: $MASTER_ADDR"
    echo "  MASTER_PORT: $MASTER_PORT"
    echo "  WORLD_SIZE: $WORLD_SIZE"
    echo "  RANK: $RANK"
else
    export MASTER_ADDR=127.0.0.1
    # export MASTER_PORT=3$(($RANDOM % 10))$(($RANDOM % 10))15
    export MASTER_PORT=6379
    export WORLD_SIZE=1
    export RANK=0

    echo "检测到Worker已设置的分布式环境变量:"
    echo "  MASTER_ADDR: $MASTER_ADDR"
    echo "  MASTER_PORT: $MASTER_PORT"
    echo "  WORLD_SIZE: $WORLD_SIZE"
    echo "  RANK: $RANK"
fi

BATCH_SIZE=$((WORLD_SIZE * 8 * NODE))


ray stop

set -x 

# export RAY_raylet_start_wait_time_s=120

if [ "$RANK" == "0" ]; then

    # Start the Ray head node
    # Wait until all WORLD_SIZE nodes have joined

    ray start --head --node-ip-address=$MASTER_ADDR --port $MASTER_PORT --num-gpus=8

    echo "Checking if GPU is ready ..."

    python3 -c "
    import time
    import ray

    required_gpus = 8
    timeout = 800

    ray.init(address='auto')
    start_time = time.time()

    while ray.available_resources().get('GPU', 0) < required_gpus:
        current_gpus = ray.available_resources().get('GPU', 0)
        print(f'Waiting for GPU resources... Required: {required_gpus}, available now: {current_gpus}')
        if time.time() - start_time > timeout:
            raise RuntimeError(
                f'Timed out waiting for GPU resources: required {required_gpus} GPUs, but only {current_gpus} are available'
            )
        time.sleep(5)
    print('GPU resources are sufficient!')
    "

    # ray status
    # sleep 30
    
    python3 -m verl.trainer.main_dapo \
        --config-path="$CONFIG_PATH" \
        --config-name='traj_grpo' \
        algorithm.adv_estimator=uis1 \
        data.train_files=./datasets/aitw_data_train_clean_final_new_local.jsonl \
        data.val_files=./datasets/aitw_data_test_clean_final_new_local.jsonl \
        data.train_batch_size=${BATCH_SIZE} \
        data.val_batch_size=$((8*BATCH_SIZE)) \
        data.max_prompt_length=8192 \
        data.max_response_length=125 \
        data.truncation='left' \
        data.num_image_limit=${IMAGE} \
        actor_rollout_ref.model.path=/cpfs01/HithinkOmniSSD/user_workspace/AITW_SFT/${CHECKPOINT}/merge/checkpoint-${CHECKPOINT_STEP} \
        actor_rollout_ref.actor.optim.lr=${LR} \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
        actor_rollout_ref.actor.use_fixed_num_mini_batches=True \
        actor_rollout_ref.actor.fixed_num_mini_batches=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS} \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=${ENTROPY} \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.max_model_len=32768 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.limit_images=${IMAGE} \
        actor_rollout_ref.rollout.n=${ROLLOUT} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        algorithm.use_kl_in_reward=False \
        algorithm.gamma=$GAMMA \
        algorithm.uis1.step_advantage_w=1.0 \
        algorithm.uis1.mode=$mode \
        algorithm.patch_threshold=$PATCH_THRESHOLD \
        algorithm.filter_groups.enable=$DAPO \
        algorithm.filter_groups.metric='seq_future_reward' \
        algorithm.filter_groups.std_threshold=$DAPO_THRESHOLD \
        algorithm.filter_groups.max_num_gen_batches=0 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.default_local_dir=/cpfs01/HithinkOmniSSD/user_workspacecheckpoint/gui_traj_grpo/${EXPERIMENT_NAME} \
        trainer.project_name='gui_traj_grpo' \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.val_before_train=True \
        trainer.total_epochs=8 $@

# ray stop  
else
    
    wake 40
    
    # Keep trying until head is available
    until ray start --address="$MASTER_ADDR:6379" --num-gpus=8; do
      echo "[worker] Waiting for head at $MASTER_ADDR:6379 ..."
      sleep 5
    done
    
    # Continue check cluster status
    while ray status >/dev/null 2>&1; do
        wake 10
    done
    echo "Ray cluster stopped, exiting worker script"
    exit 0
fi

wake 86000

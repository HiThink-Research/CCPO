# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
set -x

nvidia-smi

# 1) light env
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export HYDRA_FULL_ERROR=0

# Ray配置 - 增加心跳超时和重试设置
export RAY_raylet_heartbeat_timeout_milliseconds=60000
export RAY_gcs_rpc_server_reconnect_timeout_s=300
export RAY_raylet_start_wait_time_s=120

export WANDB_API_KEY="3714377978ed0839cf33b00761df4c027c19ad21"

# pip install -e .
# pip install vllm==0.9.2
# pip install transformers==4.52.3
# pip install flash-attn==2.7.4.post1 --no-build-isolation

shopt -s expand_aliases
alias wake='sleep'


# rlog
# source /opt/conda/etc/profile.d/conda.sh
export PYTHONPATH=.
export VLLM_USE_V1=1

# Optional: Set wandb project and entity
# export WANDB_PROJECT="DAPO-AC"

# Optional: Set wandb mode (online/offline)
export WANDB_MODE="online"  # or "offline" for offline logging

# Set GLOO_SOCKET_IFNAME from NCCL_SOCKET_IFNAME if not already set
if [ -n "$NCCL_SOCKET_IFNAME" ] && [ -z "$GLOO_SOCKET_IFNAME" ]; then
    export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
fi

ENGINE=${1:-vllm}
mode="mean_std_norm"
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"
#CONFIG_PATH="/data/config"

GAMMA=0.5
DAPO=True
DAPO_THRESHOLD=0.3
PATCH_THRESHOLD=2

EXPERIMENT_NAME="qwenvl_uis1_DAPO_${DAPO}_${DAPO_THRESHOLD}_patch_${PATCH_THRESHOLD}_gamma_${GAMMA}_fast_CCPO_3AO_2_nodes_noterm_nothink_lora_all_3000_MAX"
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
    export MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
    export WORLD_SIZE=1
    export RANK=0

    echo "检测到Worker已设置的分布式环境变量:"
    echo "  MASTER_ADDR: $MASTER_ADDR"
    echo "  MASTER_PORT: $MASTER_PORT"
    echo "  WORLD_SIZE: $WORLD_SIZE"
    echo "  RANK: $RANK"
fi

BATCH_SIZE=$((WORLD_SIZE * 8))


ray stop

set -x 

# export RAY_raylet_start_wait_time_s=120

if [ "$RANK" == "0" ]; then

    # 启动Ray头节点
    # Wait until all WORLD_SIZE nodes have joined

    ray start --head --node-ip-address=$MASTER_ADDR --num-gpus=8

    echo "等待Ray头节点完全启动..."
    sleep 10

    echo "检查GPU资源和节点是否就绪..."
    
    python3 -c "
import time
import ray

required_gpus = $((WORLD_SIZE * 8))  # 动态计算需要的GPU数量
required_nodes = $WORLD_SIZE  # 需要的节点数量
timeout = 300  # 5分钟超时
check_interval = 3  # 每3秒检查一次

ray.init(address='auto', ignore_reinit_error=True)
start_time = time.time()

print(f'等待集群就绪: 需要 {required_nodes} 个节点, {required_gpus} 个GPU')

while True:
    try:
        # 获取可用资源
        available_resources = ray.available_resources()
        current_gpus = available_resources.get('GPU', 0)
        
        # 获取集群状态
        cluster_resources = ray.cluster_resources()
        total_gpus = cluster_resources.get('GPU', 0)
        
        # 尝试获取节点信息
        try:
            nodes = ray.nodes()
            alive_nodes = [n for n in nodes if n.get('Alive', False)]
            num_alive_nodes = len(alive_nodes)
        except:
            num_alive_nodes = 1  # 至少头节点是活着的
        
        print(f'集群状态: 存活节点={num_alive_nodes}/{required_nodes}, 总GPU={total_gpus}/{required_gpus}, 可用GPU={current_gpus}/{required_gpus}')
        
        # 检查是否满足要求
        if num_alive_nodes >= required_nodes and total_gpus >= required_gpus and current_gpus >= required_gpus:
            print('✓ 集群资源已满足要求！')
            print(f'  存活节点: {num_alive_nodes}')
            print(f'  总GPU: {total_gpus}')
            print(f'  可用GPU: {current_gpus}')
            break
        
        # 检查超时
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise RuntimeError(f'集群资源等待超时 ({timeout}秒): 需要 {required_nodes} 节点/{required_gpus} GPU, 当前 {num_alive_nodes} 节点/{total_gpus} GPU')
        
        time.sleep(check_interval)
    except Exception as e:
        print(f'检查资源时出错: {e}')
        time.sleep(check_interval)
"

    # ray status
    # sleep 30
    
    python3 -m verl.trainer.main_dapo \
        --config-path="$CONFIG_PATH" \
        --config-name='traj_grpo' \
        algorithm.adv_estimator=uis1 \
        data.train_files=./datasets/android_control_train_clean_final_noterm.jsonl \
        data.val_files=./datasets/android_control_evaluation_fixed.jsonl \
        data.train_batch_size=${BATCH_SIZE} \
        data.val_batch_size=$((8*BATCH_SIZE)) \
        data.max_prompt_length=16384 \
        data.max_response_length=256 \
        data.truncation='left' \
        data.num_image_limit=4 \
        actor_rollout_ref.model.path=/mnt/HithinkOmniSSD/user_workspace/songyurun/AC_SFT/7B_1AO_bz8_sequence_baseline_lora_all/merge/checkpoint-3000 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
        actor_rollout_ref.actor.use_fixed_num_mini_batches=True \
        actor_rollout_ref.actor.fixed_num_mini_batches=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
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
        actor_rollout_ref.rollout.limit_images=4 \
        actor_rollout_ref.rollout.n=4 \
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
        trainer.default_local_dir=/mnt/workspace/checkpoint/gui_traj_grpo/${EXPERIMENT_NAME} \
        trainer.project_name='gui_traj_grpo' \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.val_before_train=True \
        trainer.total_epochs=3 $@
    ray stop
else
    
    echo "[worker] 等待主节点Ray集群启动..."
    # 减少初始等待时间，改为更智能的重试逻辑
    sleep 15
    
    # 先检查主节点是否可访问
    echo "[worker] 检查主节点连接..."
    max_retries=60
    retry_count=0
    connected=false
    
    while [ $retry_count -lt $max_retries ] && [ "$connected" = false ]; do
        # 尝试连接Ray集群
        if ray start --address="$MASTER_ADDR:6379" --num-gpus=8 > /tmp/ray_start.log 2>&1; then
            # 检查输出中是否包含成功信息
            if grep -q "Successfully connected\|Ray runtime started\|Connected to Ray cluster" /tmp/ray_start.log 2>/dev/null; then
                echo "[worker] ✓ 成功连接到Ray集群"
                connected=true
                break
            fi
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            echo "[worker] ✗ 连接Ray集群失败，已达到最大重试次数 ($max_retries)"
            echo "[worker] 最后的错误信息:"
            cat /tmp/ray_start.log 2>/dev/null || echo "无错误日志"
            exit 1
        fi
        echo "[worker] 等待主节点Ray集群启动... (重试 $retry_count/$max_retries)"
        sleep 5
    done
    
    if [ "$connected" = false ]; then
        echo "[worker] ✗ 无法连接到Ray集群"
        exit 1
    fi
    
    # 验证连接并等待GPU注册
    echo "[worker] 验证GPU资源注册..."
    python3 -c "
import time
import ray

max_wait = 120  # 最多等待2分钟
start_time = time.time()

try:
    ray.init(address='$MASTER_ADDR:6379', ignore_reinit_error=True)
    
    while time.time() - start_time < max_wait:
        try:
            cluster_resources = ray.cluster_resources()
            my_gpus = cluster_resources.get('GPU', 0)
            print(f'[worker] 集群总GPU: {my_gpus}')
            
            # 检查是否至少有16个GPU（2节点×8GPU）
            if my_gpus >= 16:
                print('[worker] ✓ GPU资源已注册')
                break
            time.sleep(3)
        except Exception as e:
            print(f'[worker] 检查资源时出错: {e}')
            time.sleep(3)
    else:
        print('[worker] ⚠ GPU资源注册超时，但继续执行')
except Exception as e:
    print(f'[worker] ⚠ 初始化Ray时出错: {e}，但继续执行')
"
    
    # 持续检测集群状态
    echo "[worker] 监控Ray集群状态..."
    while ray status >/dev/null 2>&1; do
        wake 10
    done
    echo "[worker] Ray集群已停止，退出worker脚本"
    exit 0
fi

wake 86000

#!/bin/bash
#SBATCH --job-name=qwen35_397b_32n
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=logs/qwen35_397b_%j.out
#SBATCH --error=logs/qwen35_397b_%j.err


# ============================================================================
# SLURM Multi-Node Training Script - Qwen3.5-397B-A17B (32 nodes, 256 GPUs)
# ============================================================================
# Usage: sbatch slurm_train_qwen35_397b_32node.sbatch
#
# Requirements: 32 nodes, 256 GPUs total (8 per node)
# Model parallelism: TP=2, PP=4, EP=32, DP=32
#   World = TP(2) × PP(4) × DP(32) = 256 GPUs
#   EP=32 fully shards MoE experts across all 32 GPUs in each DP group
# ============================================================================

# Print job information
echo "================================================================"
echo "Qwen3.5-397B-A17B Pretraining — 32 Nodes / 256 GPUs"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 8))"
echo "Node list: $SLURM_JOB_NODELIST"
echo "================================================================"

export WANDB_MODE=offline

# ============================================================================
# System Limits Configuration
# ============================================================================
ulimit -n unlimited 2>/dev/null || ulimit -n 1048576
ulimit -u unlimited 2>/dev/null || ulimit -u 256000
ulimit -s unlimited 2>/dev/null || ulimit -s 65536
ulimit -l unlimited 2>/dev/null || ulimit -l unlimited
ulimit -c unlimited 2>/dev/null || true

echo "System limits:"
echo "  Max open files: $(ulimit -n)"
echo "  Max processes: $(ulimit -u)"
echo "  Stack size: $(ulimit -s)"
echo "  Memory lock: $(ulimit -l)"
echo "================================================================"

# ============================================================================
# NCCL Configuration
# ============================================================================
export NCCL_DEBUG=WARN  # Set to INFO for debugging, WARN for production

# ============================================================================
# Network Interface Detection
# Set USE_INFINIBAND=1 to enable InfiniBand, 0 for Ethernet only
# ============================================================================
USE_INFINIBAND=${USE_INFINIBAND:-1}

if [ "$USE_INFINIBAND" -eq 1 ]; then
    # InfiniBand Configuration for mlx5 devices
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9
    export NCCL_IB_GID_INDEX=3
    export NCCL_NET_GDR_LEVEL=5
    export NCCL_IB_TIMEOUT=22
    export NCCL_IB_RETRY_CNT=7

    IB_INTERFACE=$(ip link show 2>/dev/null | grep -o "ib[0-9]" | head -1)
    if [ -n "$IB_INTERFACE" ]; then
        export NCCL_SOCKET_IFNAME=$IB_INTERFACE
    fi
else
    # Ethernet Configuration (fallback)
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=eth0
fi

# NCCL Tuning for High-Speed InfiniBand
export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_NCHANNELS=32
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=8

# ============================================================================
# Distributed Training Configuration
# ============================================================================
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * 8))  # 32 nodes × 8 GPUs = 256
export GPUS_PER_NODE=8

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Network: $([ "$USE_INFINIBAND" -eq 1 ] && echo "InfiniBand" || echo "Ethernet")"
echo "Model: Qwen3.5-397B-A17B (TP=2, PP=4, EP=32, DP=32 on 256 GPUs)"
echo "================================================================"

# ============================================================================
# Container Configuration
# ============================================================================
CONTAINER_IMAGE="/mnt/shared/images/megatron-bridge-transformers-5.3.0.dev0.sqsh"
CONTAINER_NAME="qwen35_397b_train_32n_${SLURM_JOB_ID}"
MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:-$(pwd)}"
CONTAINER_MOUNT_PATH="/megatron-bridge"

CONTAINER_MOUNTS="${MEGATRON_BRIDGE_PATH}:${CONTAINER_MOUNT_PATH},tmpfs:/dev/shm:size=512G"

# ============================================================================
# Python Environment
# ============================================================================
export PYTHONPATH="${PYTHONPATH}:${CONTAINER_MOUNT_PATH}/3rdparty/Megatron-LM/megatron"
export PYTHONUNBUFFERED=1

# ============================================================================
# Create log directory
# ============================================================================
mkdir -p logs

# ============================================================================
# Launch training across all nodes
# ============================================================================
echo "Launching Qwen3.5-397B-A17B training on 32 nodes..."
echo "================================================================"

srun --container-image="${CONTAINER_IMAGE}" \
     --container-name="${CONTAINER_NAME}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     --container-workdir="${CONTAINER_MOUNT_PATH}" \
     --container-remap-root \
     --container-writable \
     --mpi=pmix \
     bash run_training_container_qwen35_397b_32node.sh

exit_code=$?
echo "================================================================"
if [ $exit_code -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $exit_code"
fi
echo "================================================================"
exit $exit_code

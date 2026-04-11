#!/bin/bash
# ============================================================================
# Container Training Execution Script - Qwen3.5-397B-A17B (32 nodes / 256 GPUs)
# ============================================================================
# This script runs inside the container on each node.
# Called by srun from slurm_train_qwen35_397b_32node.sbatch
# ============================================================================

set -e
set -u

echo "================================================================"
echo "Starting Qwen3.5-397B-A17B training on node: $(hostname)"
echo "Node rank: ${SLURM_NODEID} | Local rank: ${SLURM_LOCALID}"
echo "================================================================"

ulimit -n unlimited 2>/dev/null || ulimit -n 1048576
ulimit -u unlimited 2>/dev/null || ulimit -u 256000
ulimit -s unlimited 2>/dev/null || ulimit -s 65536
ulimit -l unlimited 2>/dev/null || ulimit -l unlimited
ulimit -c unlimited 2>/dev/null || true

# ============================================================================
# Environment Setup
# ============================================================================
source /opt/venv/bin/activate
export PYTHONPATH="/megatron-bridge/src:/megatron-bridge/3rdparty/Megatron-LM"
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# NVSHMEM Configuration for DeepEP
# ============================================================================
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_HCA_LIST="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_SYMMETRIC_SIZE=4294967296  # 4GB

export HF_DATASETS_CACHE="/megatron-bridge/cache_hf_dataset"

# Pre-download dataset on node 0, other nodes wait, then all read from cache
DOWNLOAD_MARKER="${HF_DATASETS_CACHE}/.download_complete_${SLURM_JOB_ID}"

if [ "${SLURM_NODEID}" -eq 0 ]; then
    echo "Node 0: Pre-downloading dataset to ${HF_DATASETS_CACHE}..."
    mkdir -p "${HF_DATASETS_CACHE}"
    python -c "
from datasets import load_dataset
for split in ['train', 'validation', 'test']:
    try:
        load_dataset('naver-clova-ix/cord-v2', split=split)
        print(f'  Downloaded split: {split}')
    except Exception as e:
        print(f'  Skipped split {split}: {e}')
"
    touch "${DOWNLOAD_MARKER}"
    echo "Node 0: Dataset download complete."
else
    echo "Node ${SLURM_NODEID}: Waiting for dataset download..."
    while [ ! -f "${DOWNLOAD_MARKER}" ]; do sleep 1; done
    echo "Node ${SLURM_NODEID}: Dataset ready."
fi

export HF_DATASETS_OFFLINE=1

# ============================================================================
# Distributed Training Variables
# ============================================================================
export NODE_RANK=${SLURM_NODEID}
export LOCAL_RANK=${SLURM_LOCALID}
export RANK=$((NODE_RANK * GPUS_PER_NODE + LOCAL_RANK))

export WANDB_MODE=offline

# ============================================================================
# Launch Training with torchrun
# ============================================================================
if [ "${SLURM_LOCALID}" -eq 0 ]; then
    python -m torch.distributed.run \
        --nnodes=${SLURM_JOB_NUM_NODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        pretrain_qwen35_vl_397b_32node.py

    exit_code=$?
    echo "================================================================"
    echo "Node ${NODE_RANK} finished with exit code: ${exit_code}"
    echo "================================================================"
    exit ${exit_code}
else
    exit 0
fi

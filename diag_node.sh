#!/bin/bash
# Node diagnostic script - run on specific worker nodes to compare performance
# Usage: sbatch --nodelist=worker-XX diag_node.sh

#SBATCH --job-name=node_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=logs/diag_%j_%N.out

CONTAINER_IMAGE="/mnt/shared/images/megatron-bridge-transformers-5.3.0.dev0.sqsh"
MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:-$(pwd)}"

mkdir -p logs

echo "========================================"
echo "Node Diagnostic: $(hostname)"
echo "Date: $(date)"
echo "========================================"

echo ""
echo "=== 1. CPU Info ==="
lscpu | grep -E "Model name|CPU\(s\)|Thread|MHz|CPU max|CPU min|NUMA"

echo ""
echo "=== 2. CPU Frequency (current) ==="
cat /proc/cpuinfo | grep "cpu MHz" | sort -t: -k2 -n | head -4
echo "..."
cat /proc/cpuinfo | grep "cpu MHz" | sort -t: -k2 -n | tail -4

echo ""
echo "=== 3. Memory Info ==="
free -g

echo ""
echo "=== 4. GPU Hardware ==="
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,temperature.gpu,clocks.current.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader

echo ""
echo "=== 5. IB Status ==="
ibstat 2>/dev/null | grep -E "CA |Port |State|Rate" | head -20

echo ""
echo "=== 6. Running torch.compile benchmark inside container ==="
srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${MEGATRON_BRIDGE_PATH}:/megatron-bridge" \
     --container-workdir="/megatron-bridge" \
     --container-remap-root \
     --container-writable \
     bash -c '
source /opt/venv/bin/activate
python3 -c "
import time
import torch
import torch._dynamo

print(\"PyTorch version:\", torch.__version__)
print(\"CUDA version:\", torch.version.cuda)
print(\"GPU:\", torch.cuda.get_device_name(0))
print()

# Benchmark 1: Simple matmul compile
def matmul_fn(a, b):
    return torch.matmul(a, b)

a = torch.randn(4096, 4096, device=\"cuda\", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device=\"cuda\", dtype=torch.bfloat16)

print(\"=== Benchmark: torch.compile matmul 4096x4096 ===\")
t0 = time.time()
compiled_fn = torch.compile(matmul_fn)
_ = compiled_fn(a, b)
torch.cuda.synchronize()
t1 = time.time()
print(f\"  First call (compile + run): {t1-t0:.2f}s\")

_ = compiled_fn(a, b)
torch.cuda.synchronize()
t2 = time.time()
print(f\"  Second call (cached): {t2-t1:.4f}s\")

# Benchmark 2: MLP-like compile (closer to real model)
class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up = torch.nn.Linear(d, 4*d, bias=False)
        self.gate = torch.nn.Linear(d, 4*d, bias=False)
        self.down = torch.nn.Linear(4*d, d, bias=False)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

print()
print(\"=== Benchmark: torch.compile MLP d=4096 ===\")
mlp = MLP(4096).cuda().bfloat16()
x = torch.randn(2, 4096, 4096, device=\"cuda\", dtype=torch.bfloat16)

t0 = time.time()
compiled_mlp = torch.compile(mlp)
_ = compiled_mlp(x)
torch.cuda.synchronize()
t1 = time.time()
print(f\"  First call (compile + run): {t1-t0:.2f}s\")

_ = compiled_mlp(x)
torch.cuda.synchronize()
t2 = time.time()
print(f\"  Second call (cached): {t2-t1:.4f}s\")

# Benchmark 3: CPU-only compilation speed (no GPU)
print()
print(\"=== Benchmark: CPU compilation speed (no GPU) ===\")
torch._dynamo.reset()
def cpu_fn(x):
    for _ in range(10):
        x = x * 2 + 1
    return x

cx = torch.randn(1024, 1024)
t0 = time.time()
compiled_cpu = torch.compile(cpu_fn, backend=\"eager\")
_ = compiled_cpu(cx)
t1 = time.time()
print(f\"  eager backend compile: {t1-t0:.2f}s\")

# Benchmark 4: Raw CPU speed
print()
print(\"=== Benchmark: Raw CPU speed (numpy-like) ===\")
import numpy as np
arr = np.random.randn(8192, 8192).astype(np.float32)
t0 = time.time()
for _ in range(3):
    _ = np.dot(arr, arr)
t1 = time.time()
print(f\"  3x matmul 8192x8192 fp32: {t1-t0:.2f}s\")

print()
print(\"=== Done ===\")
"
'

echo ""
echo "========================================"
echo "Diagnostic complete: $(hostname)"
echo "========================================"

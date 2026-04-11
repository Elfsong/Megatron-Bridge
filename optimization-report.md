# Qwen3.5-397B-A17B Training Optimization Report

**Date:** 2026-04-10 ~ 2026-04-11
**Hardware:** 32 nodes x 8 H200 GPUs (256 GPUs total, 141 GB HBM3e each)
**Network:** 400 Gbps InfiniBand (8x ConnectX-7 per node)
**Framework:** Megatron-Bridge + Megatron-LM, NeMo 25.11 container

---

## Final Result

| Metric | Baseline | Optimized | Improvement |
|---|---|---|---|
| Step time | 63.5s | 42.5s | **-33%** |
| TFLOP/s/GPU | 56.4 | 84.0 | **+49%** |
| Tokens/s/GPU | 516 | 773 | **+50%** |
| Total throughput | 132K tok/s | 198K tok/s | **+50%** |

---

## Trial Summary

### Trial 1: BF16 EP=32 Baseline (Job 8255)
**Config:** TP=2, PP=4, EP=32, DP=32, MBS=1, GBS=2048, seq_len=4096
**Result:** 63.5s/step, 56.4 TFLOP/s, **516 tok/s/GPU**
**Notes:** First successful run after fixing container lock (Job 8252), `is_flash_attn` import (Job 8253), and `diffusers` import (Job 8254) errors.

### Trial 2: MBS=2 (Job 8256, 8257)
**Config:** Same as baseline but MBS=2
**Result:** FAILED
- Job 8256: NCCL timeout due to worker-56 CPU frequency anomaly (200 MHz)
- Job 8257: OOM on PP stage 0 (134 GB / 141 GB limit)
**Conclusion:** H200 141 GB insufficient for MBS=2 on PP stage 0.

### Trial 3: FP8 Hybrid EP=32 (Job 8261)
**Config:** Added `fp8="hybrid"` (tensorwise recipe)
**Result:** 62.2s/step, 57.5 TFLOP/s, **527 tok/s/GPU** (+2%)
**Notes:** Marginal speed improvement. FP8 saved ~12 GB/GPU memory but bottleneck was communication, not compute.

### Trial 4: FP8 EP=16 + VPP=2 (Job 8263)
**Config:** EP 32->16, VPP=2
**Result:** FAILED - `num_layers_per_pipeline_rank (15) not divisible by vp_size (2)`

### Trial 5: FP8 EP=16 + VPP=3 (Job 8264)
**Config:** EP=16, VPP=3
**Result:** FAILED - OOM (PP stage 0 peaked at 118 GB/GPU + NCCL buffers exceeded 141 GB)
**Notes:** VPP increases activation peak due to multiple virtual chunks in flight.

### Trial 6: FP8 EP=16 (Job 8265) -- BEST CONFIG
**Config:** EP 32->16, FP8 hybrid, no VPP
**Result:** 45.5s -> 42.5s/step, 78-84 TFLOP/s, **720-773 tok/s/GPU** (+40-50%)
**Notes:** EP reduction halved all-to-all communication volume. Single biggest optimization. Speed continued improving throughout training (45s at iter 30 -> 42.5s at iter 400+).

### Trial 7: FP8 EP=16 + delay_wgrad_compute (Job 8266)
**Config:** Added `delay_wgrad_compute=True`
**Result:** ~50s/step, ~700 tok/s/GPU (no improvement)
**Conclusion:** Wgrad compute overlap ineffective with MBS=1 (computation too small to overlap meaningfully).

### Trial 8: Blockwise FP8 (Jobs 8267-8269)
**Config:** `fp8="e4m3"`, `fp8_recipe="blockwise"`, `fp8_param_gather=True`
**Result:** FAILED
- Job 8267: `str has no attribute grad_reduce_in_fp32` (MixedPrecisionConfig type error)
- Job 8268: `fp8 not set` (FP8 settings overwritten by MixedPrecisionConfig.setup())
- Job 8269: `Unsupported quantizer for Userbuffers` (blockwise FP8 incompatible with `tp_comm_overlap`)
**Conclusion:** Blockwise FP8 cannot be used with TP communication overlap on current Megatron/TE version.

### Trial 9: grad_reduce_in_fp32=False (Job 8270)
**Config:** Custom MixedPrecisionConfig with `grad_reduce_in_fp32=False`
**Result:** ~49s early, ~48s late (vs ~45s for True)
- Early iterations: 3% faster than True
- Late iterations: 5% slower than True
- Net effect: **negative**
**Root cause investigation:** Discovered `MixedPrecisionConfig.setup()` silently overrides `ddp.grad_reduce_in_fp32` with its own default (`True`). Fixed by creating explicit `MixedPrecisionConfig` object.
**Conclusion:** BF16 grad reduce introduces precision loss that affects late-stage optimization efficiency.

---

## Infrastructure Findings

### worker-56 CPU Frequency Anomaly
- **Issue:** 6 CPU cores stuck at 199-200 MHz (below configured minimum of 800 MHz)
- **Impact:** torch.compile 10x slower, causing NCCL timeout in multi-node training
- **Evidence:** Diagnostic benchmark showed 26.62s compile time vs 2.52s on healthy node
- **Root cause:** CPU P-state control failure (governor set to `performance` but cores below min frequency)
- **Workaround:** `#SBATCH --exclude=worker-56`
- **Status:** Report filed for admin review

---

## Configuration Evolution

```
Baseline:  TP=2, PP=4, EP=32, BF16          -> 63.5s, 516 tok/s/GPU
+ FP8:     TP=2, PP=4, EP=32, FP8 hybrid    -> 62.2s, 527 tok/s/GPU  (+2%)
+ EP=16:   TP=2, PP=4, EP=16, FP8 hybrid    -> 42.5s, 773 tok/s/GPU  (+50%)
```

---

## What Worked

| Optimization | Impact | Why |
|---|---|---|
| EP 32 -> 16 | **+40-50%** | Halved all-to-all communication across 32 nodes |
| FP8 hybrid | +2% speed, -12 GB memory | Faster GEMM + memory savings |
| Exclude worker-56 | Prevents NCCL timeout | CPU frequency anomaly |

## What Didn't Work

| Optimization | Result | Why |
|---|---|---|
| MBS 1 -> 2 | OOM | PP stage 0 exceeds 141 GB |
| VPP=3 | OOM | Multiple virtual chunks increase activation peak |
| delay_wgrad_compute | No effect | MBS=1 wgrad too small to overlap |
| Blockwise FP8 | Incompatible | Userbuffers don't support blockwise quantizer |
| grad_reduce_in_fp32=False | Slower (-5%) | Precision loss hurts late-stage optimization |

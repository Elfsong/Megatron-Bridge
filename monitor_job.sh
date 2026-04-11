#!/bin/bash
# Usage: ./monitor_job.sh <JOB_ID> [interval_seconds]
# Monitors GPU utilization (all nodes) and token throughput per GPU.

set -u

JOB_ID=${1:?Usage: ./monitor_job.sh <JOB_ID> [interval_seconds]}
INTERVAL=${2:-90}
LOG_DIR="$(dirname "$0")/logs"
LOG_FILE="${LOG_DIR}/qwen35_397b_${JOB_ID}.out"
ERR_FILE="${LOG_DIR}/qwen35_397b_${JOB_ID}.err"

# Constants for throughput calculation
SEQ_LEN=4096
NUM_GPUS=256

while true; do
  state=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)
  if [ -z "$state" ]; then
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') | Job $JOB_ID ended ==="
    tail -5 "$LOG_FILE" 2>/dev/null
    break
  fi

  echo "================================================================"
  echo "  $(date '+%Y-%m-%d %H:%M:%S') | Job $JOB_ID | State: $state"
  echo "================================================================"

  # --- Token Throughput ---
  last_iter=$(grep "iteration " "$LOG_FILE" 2>/dev/null | tail -1)
  if [ -n "$last_iter" ]; then
    iter=$(echo "$last_iter" | grep -oP 'iteration\s+\K\d+')
    total=$(echo "$last_iter" | grep -oP 'iteration\s+\d+/\s*\K\d+')
    gbs=$(echo "$last_iter" | grep -oP 'global batch size:\s+\K\d+')
    step_ms=$(echo "$last_iter" | grep -oP 'elapsed time per iteration \(ms\):\s+\K[0-9.]+')
    loss=$(echo "$last_iter" | grep -oP 'lm loss:\s+\K[0-9.E+-]+')
    grad=$(echo "$last_iter" | grep -oP 'grad norm:\s+\K[0-9.]+')

    if [ -n "$step_ms" ] && [ -n "$gbs" ]; then
      throughput=$(awk "BEGIN {
        step_s = $step_ms / 1000
        tok_per_step = $gbs * $SEQ_LEN
        tok_sec = tok_per_step / step_s
        tok_sec_gpu = tok_sec / $NUM_GPUS
        printf \"%.0f tokens/sec total | %.0f tokens/sec/GPU\", tok_sec, tok_sec_gpu
      }")
      pct=$(awk "BEGIN { printf \"%.1f\", $iter / $total * 100 }")
      step_s=$(awk "BEGIN { printf \"%.1f\", $step_ms / 1000 }")

      echo ""
      echo "  Iteration:  $iter / $total ($pct%)"
      echo "  Step Time:  ${step_s}s"
      echo "  Throughput: $throughput"
      echo "  LM Loss:    $loss"
      echo "  Grad Norm:  $grad"
    fi
  else
    echo ""
    echo "  [No iterations logged yet]"
  fi

  # --- Errors ---
  errs=$(tail -10 "$ERR_FILE" 2>/dev/null | grep -iE "(error|OOM|CUDA|Traceback|killed)" | tail -3)
  if [ -n "$errs" ]; then
    echo ""
    echo "  !! ERRORS:"
    echo "$errs" | sed 's/^/    /'
  fi

  # --- GPU Utilization ---
  num_nodes=$(squeue -j "$JOB_ID" -h -o "%D" 2>/dev/null)
  if [ -n "$num_nodes" ] && [ "$num_nodes" -gt 0 ]; then
    raw=$(srun --jobid="$JOB_ID" --overlap --nodes="$num_nodes" --ntasks-per-node=1 bash -c '
      h=$(hostname -s)
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | while read line; do
        echo "$h $line"
      done
    ' 2>/dev/null)

    if [ -n "$raw" ]; then
      echo ""
      echo "$raw" | awk '{
        split($0, parts, " ")
        node=parts[1]
        rest=substr($0, length(node)+2)
        split(rest, vals, ",")
        util=vals[1]+0
        mem=vals[2]+0
        gpu_count[node]++
        total_util[node]+=util
        total_mem[node]+=mem
        all_util+=util
        all_count++
      }
      END {
        n=asorti(total_util, sorted)
        printf "  %-14s | AvgUtil | Mem(GB)\n", "Node"
        printf "  %-14s-|---------|--------\n", "--------------"
        for(i=1;i<=n;i++) {
          nd=sorted[i]
          c=gpu_count[nd]
          printf "  %-14s | %5.0f%% | %d\n", nd, total_util[nd]/c, total_mem[nd]/1024
        }
        printf "  %-14s-|---------|--------\n", "--------------"
        printf "  %-14s | %5.0f%% |\n", "CLUSTER AVG", all_util/all_count
      }'
    fi
  fi

  echo ""
  sleep "$INTERVAL"
done

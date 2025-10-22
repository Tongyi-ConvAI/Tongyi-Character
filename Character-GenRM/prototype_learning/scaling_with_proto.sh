#!/bin/bash
set -euo pipefail

# ========= Argument Check =========
if [ $# -lt 1 ]; then
  echo "Usage: $0 <chatbot|prism>"
  exit 1
fi

MODE="$1"
if [ "$MODE" != "chatbot" ] && [ "$MODE" != "prism" ]; then
  echo "Error: Mode must be either chatbot_arena or prism"
  exit 1
fi

# ========= Environment =========
export TORCH_SHOW_CPP_STACKTRACES=1
ulimit -n 65535 || true

# ========= Fixed Parameters =========
FEWSHOTS=3
VLLM_PORT=8000
VLLM_GPUS=8
VLLM_DEVICES="0,1,2,3,4,5,6,7"

# ========= Global State =========
VLLM_PID=""
CLEANUP_DONE=0

# ========= Task Configuration =========
declare -a TASK_MODEL_PATHS
declare -a TASK_SERVED_NAMES

if [ "$MODE" == "chatbot" ]; then
  # ✅ Chatbot mode
  TASK_MODEL_PATHS+=(../checkpoints/P-GenRM-8B-ChatbotArena)
  TASK_SERVED_NAMES+=(Chatbot_Arena_test_time_user_based_scaling)
else
  # ✅ Prism mode
  TASK_MODEL_PATHS+=(../checkpoints/P-GenRM-8B-PRISM)
  TASK_SERVED_NAMES+=(PRISM_test_time_user_based_scaling)
fi

# ========= Cleanup Function =========
trap cleanup EXIT INT TERM
cleanup() {
  if [ "$CLEANUP_DONE" -eq 1 ]; then return; fi
  echo "---"
  echo "Running cleanup..."
  if [ -n "${VLLM_PID}" ]; then
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "Stopping vLLM service (PID: ${VLLM_PID})..."
      kill "${VLLM_PID}" || true
      sleep 2
    else
      echo "vLLM service (PID: ${VLLM_PID}) no longer exists."
    fi
  fi
  echo "Cleanup complete."
  CLEANUP_DONE=1
}

# ========= Wait for vLLM Ready =========
wait_for_vllm() {
  local url=$1
  local pid=$2
  local timeout=${3:-1800}
  local elapsed=0

  printf "Waiting for vLLM service to be ready (URL: %s, PID: %s) " "$url" "$pid"
  while true; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo
      echo "✗ Error: vLLM process (PID: $pid) terminated unexpectedly!" >&2
      exit 1
    fi
    if curl -s --connect-timeout 2 "$url/v1/models" > /dev/null; then
      echo "✓ Service is ready!"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed+5))
    printf "."
    if [ $elapsed -ge $timeout ]; then
      echo
      echo "✗ Timeout waiting for vLLM (${timeout}s)" >&2
      exit 1
    fi
  done
}

# ========= Run a Single Evaluation Task =========
run_evaluation_task() {
  local model_path="$1"
  local served_model_name="$2"
  local log_file="${served_model_name}.txt"

  VLLM_PID=""
  CLEANUP_DONE=0

  echo "========================================================================"
  echo "===> Starting Task: ${served_model_name} (Mode: ${MODE})"
  echo "========================================================================"
  echo "Model path: ${model_path}"
  echo "Log file: ${log_file}"

  echo "Starting vLLM service (using ${VLLM_GPUS} GPUs on port ${VLLM_PORT})..."
  CUDA_VISIBLE_DEVICES=${VLLM_DEVICES} \
  python -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --served-model-name "${served_model_name}" \
    --tensor-parallel-size ${VLLM_GPUS} \
    --max-model-len 6144 \
    --max-num-batched-tokens 6144 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.55 \
    --enable-chunked-prefill \
    --port ${VLLM_PORT} \
    > "${log_file}" 2>&1 &

  VLLM_PID=$!
  echo "vLLM started, PID: ${VLLM_PID}"

  wait_for_vllm "http://127.0.0.1:${VLLM_PORT}" "${VLLM_PID}"

  echo "Starting evaluation task..."
  python -u ./scaling_with_proto.py \
    --model "${served_model_name}" \
    --fs ${FEWSHOTS} \
    --max-workers 20 \
    --dataset "${MODE}" \
    --proto-save-dir "./saved_pt_0.6B/user_history_train_${MODE}/k_100" \
    --st-embed ../../Qwen3-Embedding-0.6B \
    --dataset-csv "./${MODE}/user_history_train_${MODE}.csv" \
    --examples-col examples_analysis_process \
    --neighbors 8 \
    --coef-self 0.5 \
    --coef-neighbors 0.5 \
    --use-trained \
    --verbose \
    --temperature 1.0 \
    --self-runs 16

  if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Shutting down vLLM service..."
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
  fi
  echo "Task ${served_model_name} completed."
}

# ========= Main Scheduler =========
for i in "${!TASK_MODEL_PATHS[@]}"; do
  run_evaluation_task "${TASK_MODEL_PATHS[$i]}" "${TASK_SERVED_NAMES[$i]}"
done

echo "All tasks completed ✅"

#!/bin/bash

# Set environment variables (set once at the beginning of the script)
export TORCH_SHOW_CPP_STACKTRACES=1
ulimit -n 65535 || true

# --- Parse command-line arguments ---
DATASET=""
TEST_SPLIT_NAME="test"  # default: test

while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --test_split_name)
      TEST_SPLIT_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --dataset <prism|chatbotarena> [--test_split_name <test|val>]"
      exit 1
      ;;
  esac
done

if [ -z "$DATASET" ]; then
  echo "✗ Error: --dataset must be specified (e.g., --dataset prism or --dataset chatbotarena)"
  exit 1
fi

# --- Global variables ---
FEWSHOTS=6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/vllm_batch_inference.py"
VLLM_PORT=8000
VLLM_GPUS=8
VLLM_DEVICES="0,1,2,3,4,5,6,7"

VLLM_PID=""
CLEANUP_DONE=0

# --- Define model path & served name based on dataset ---
declare -a TASK_MODEL_PATHS
declare -a TASK_SERVED_NAMES

if [ "$DATASET" == "prism" ]; then
  TASK_MODEL_PATHS+=("$(dirname "$(dirname "$(realpath "$0")")")/checkpoints/P-GenRM-8B-PRISM")
  TASK_SERVED_NAMES+=("prism_vllm_batch_inference")
elif [ "$DATASET" == "chatbotarena" ]; then
  TASK_MODEL_PATHS+=("$(dirname "$(dirname "$(realpath "$0")")")/checkpoints/P-GenRM-8B-ChatbotArena")
  TASK_SERVED_NAMES+=("chatbotarena_vllm_batch_inference")
else
  echo "✗ Error: unsupported dataset '${DATASET}'. Only 'prism' or 'chatbotarena' are supported."
  exit 1
fi

# --- Cleanup function ---
trap cleanup EXIT INT TERM
cleanup() {
  if [ "$CLEANUP_DONE" -eq 1 ]; then return; fi
  echo "---"
  echo "Running cleanup..."
  if [ -n "$VLLM_PID" ]; then
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "Stopping vLLM service (PID: $VLLM_PID)..."
      kill "$VLLM_PID"
      sleep 2
    else
      echo "vLLM service (PID: $VLLM_PID) no longer exists."
    fi
  fi
  echo "Cleanup done."
  CLEANUP_DONE=1
}

# --- Wait for vLLM service ---
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
      echo "✓ Service ready!"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed+5))
    printf "."
    if [ $elapsed -ge $timeout ]; then
      echo
      echo "✗ Timeout waiting (${timeout}s)" >&2
      exit 1
    fi
  done
}

# --- Core task execution ---
run_evaluation_task() {
  local model_path="$1"
  local served_model_name="$2"
  local log_file="${served_model_name}.txt"

  VLLM_PID=""
  CLEANUP_DONE=0

  echo "========================================================================"
  echo "===> Starting task: ${served_model_name}"
  echo "========================================================================"
  echo "Model path: ${model_path}"
  echo "Log file: ${log_file}"
  echo "Test split name: ${TEST_SPLIT_NAME}"

  echo "Starting vLLM service..."
  CUDA_VISIBLE_DEVICES=${VLLM_DEVICES} \
  python -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --served-model-name "$served_model_name" \
      --tensor-parallel-size ${VLLM_GPUS} \
      --max-model-len 6144 \
      --max-num-batched-tokens 8192 \
      --max-num-seqs 256 \
      --gpu-memory-utilization 0.90 \
      --enable-chunked-prefill \
      --port ${VLLM_PORT} &
  VLLM_PID=$!
  echo "vLLM started, PID: $VLLM_PID"

  wait_for_vllm "http://127.0.0.1:${VLLM_PORT}" "$VLLM_PID"

  echo "Starting scoring task..."
  python -u "$PYTHON_SCRIPT" \
      --model "$served_model_name" \
      --fs "$FEWSHOTS" \
      --max-workers 72 \
      --dataset "$DATASET" \
      --test_split_name "$TEST_SPLIT_NAME" \
      > "$log_file" 2>&1

  if [ $? -ne 0 ]; then
      echo "✗ Error: scoring script failed. Please check log file: ${log_file}"
      exit 1
  fi

  echo "Task [${served_model_name}] scoring completed ✅"
  cleanup
  echo "------------------------------------------------------------------------"
  echo
}

# --- Main ---
set -e

NUM_TASKS=${#TASK_MODEL_PATHS[@]}
echo "Planned to execute ${NUM_TASKS} scoring tasks for dataset '${DATASET}'..."
echo

for i in "${!TASK_MODEL_PATHS[@]}"; do
  run_evaluation_task "${TASK_MODEL_PATHS[$i]}" "${TASK_SERVED_NAMES[$i]}"
done

echo "🎉 All tasks completed successfully!"

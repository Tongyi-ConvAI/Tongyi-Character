#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Common parameters
MODEL=../../Qwen3-Embedding-0.6B
BASE_SAVE_DIR=./saved_pt_0.6B

# Check if at least one CSV file path is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 path/to/file1.csv [path/to/file2.csv ...]"
  exit 1
fi

# Loop through all CSV files passed as command-line arguments
for CSV in "$@"
do
  NAME=$(basename "${CSV}" .csv)
  
  # Loop over different k values
  for K in 100
  do
    echo "===================="
    echo "Running ${NAME} with k=${K}"
    echo "===================="
    
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    torchrun --standalone --nproc_per_node=8 \
    ./prototype_learning_pipeline_tensor_members_ddp.py \
      --csv "${CSV}" \
      --save_dir "${BASE_SAVE_DIR}/${NAME}/k_${K}/" \
      --local_model_path "${MODEL}" \
      --k "${K}" \
      --epochs 1 \
      --distributed \
      --reuse_cached
  done
done

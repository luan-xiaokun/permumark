#!/usr/bin/bash

log_dir="logs"
mkdir -p "$log_dir"

num1m=1000000
num5m=5000000
num10m=10000000
num50m=50000000
num100m=100000000

generate_log_file() {
    local model_path=$1
    echo "$log_dir/$(basename "$model_path").log"
}

models=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "models/meta-llama/Llama-2-7b-hf"
    "models/meta-llama/Llama-3.1-8B"
    "models/google/gemma-7b"
    "models/mistralai/Ministral-8B-Instruct-2410"
)

for model_path in "${models[@]}"; do
  log_file=$(generate_log_file "$model_path")
  echo "Fine-tuning $model_path, logs will be saved to $log_file"
  python -u evaluation/evasion_finetune.py "$model_path" -t $num1m $num5m $num10m $num50m $num100m > "$log_file"
done

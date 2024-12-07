#!/usr/bin/bash

log_dir="logs"
mkdir -p "$log_dir"

num5m=5000000
num50m=50000000
num500m=500000000
num5b=5000000000

generate_log_file() {
    local model_path=$1
    echo "$log_dir/$(basename "$model_path").log"
}

models=(
    "models/meta-llama/Llama-2-7b-hf"
    "models/meta-llama/Llama-3.1-8B"
    "models/Qwen/Qwen1.5-7B"
    "models/Qwen/Qwen2.5-7B"
    "models/google/gemma-7b"
    "models/mistralai/Ministral-8B-Instruct-2410"
)

for model_path in "${models[@]}"; do
  log_file=$(generate_log_file "$model_path")
  echo "Fine-tuning $model_path, logs will be saved to $log_file"
  python evaluation/evasion_finetune.py "$model_path" -t $num5m $num50m $num500m $num5b > "$log_file" 2>&1
done

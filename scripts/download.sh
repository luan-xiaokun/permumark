#!/usr/bin/bash

# use the following mirror if you have network issue (e.g., in China)
export HF_ENDPOINT="https://hf-mirror.com"

# uncomment the following line if you haven't logged in
# huggingface-cli login

# download datasets
mkdir -p datasets
huggingface-cli download --repo-type dataset Salesforce/wikitext --local-dir datasets/Salesforce/wikitext

# download models
mkdir -p models

models=(
    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B"
    # "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-3.1-8B"
    "Qwen/Qwen1.5-7B"
    "Qwen/Qwen2.5-7B"
    "google/gemma-7b"
    "mistralai/Ministral-8B-Instruct-2410"
)
for model_path in "${models[@]}"; do
    huggingface-cli download --repo-type model "$model_path" --local-dir "models/$model_path"
done

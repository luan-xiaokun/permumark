#!/usr/bin/bash

# use the following mirror if you have network issue (e.g., in China)
export HF_ENDPOINT=https://hf-mirror.com

# uncomment the following line if you haven't logged in
# huggingface-cli login

# download datasets
mkdir -p datasets
huggingface-cli download --repo-type dataset Salesforce/wikitext --local-dir datasets/Salesforce/wikitext

# download models
mkdir -p models
huggingface-cli download --repo-type model meta-llama/Llama-3.2-1B --local-dir models/meta-llama/Llama-3.2-1B --local-dir-use-symlinks False
huggingface-cli download --repo-type model meta-llama/Llama-3.2-3B --local-dir models/meta-llama/Llama-3.2-3B --local-dir-use-symlinks False
huggingface-cli download --repo-type model meta-llama/Llama-2-7b-hf --local-dir models/meta-llama/Llama-2-7b-hf --local-dir-use-symlinks False
huggingface-cli download --repo-type model meta-llama/Llama-3.1-8B --local-dir models/meta-llama/Llama-3.1-8B --local-dir-use-symlinks False
huggingface-cli download --repo-type model Qwen/Qwen1.5-7B --local-dir models/Qwen/Qwen1.5-7B --local-dir-use-symlinks False
huggingface-cli download --repo-type model Qwen/Qwen2.5-7B --local-dir models/Qwen/Qwen2.5-7B --local-dir-use-symlinks False
huggingface-cli download --repo-type model google/gemma-7b --local-dir models/google/gemma-7b --local-dir-use-symlinks False
huggingface-cli download --repo-type model mistralai/Ministral-8B-Instruct-2410 --local-dir models/mistralai/Ministral-8B-Instruct-2410 --local-dir-use-symlinks False
#huggingface-cli download --repo-type model meta-llama/Llama-2-13b-hf --local-dir models/meta-llama/Llama-2-13b-hf --local-dir-use-symlinks False

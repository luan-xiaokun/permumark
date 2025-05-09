#!/bin/bash

## pruning evaluation
echo "########## llama ##########"
python evaluation/evaluation.py robustness llama --size 7b --modification pruning --pruning_amount 0.5 --pruning_method wanda -v
python evaluation/evaluation.py robustness llama --size 7b --modification pruning --pruning_amount 0.5 --pruning_method sparse-gpt -v
python evaluation/evaluation.py robustness llama --size 7b --modification pruning --pruning_amount 0.7 --pruning_method wanda -v
python evaluation/evaluation.py robustness llama --size 7b --modification pruning --pruning_amount 0.7 --pruning_method sparse-gpt -v
python evaluation/evaluation.py robustness llama --size 8b --modification pruning --pruning_amount 0.5 --pruning_method wanda -v
python evaluation/evaluation.py robustness llama --size 8b --modification pruning --pruning_amount 0.5 --pruning_method sparse-gpt -v
python evaluation/evaluation.py robustness llama --size 8b --modification pruning --pruning_amount 0.7 --pruning_method wanda -v
python evaluation/evaluation.py robustness llama --size 8b --modification pruning --pruning_amount 0.7 --pruning_method sparse-gpt -v

echo "########## qwen ##########"
python evaluation/evaluation.py robustness qwen --size 7b --modification pruning --pruning_amount 0.5 --pruning_method wanda -v
python evaluation/evaluation.py robustness qwen --size 7b --modification pruning --pruning_amount 0.5 --pruning_method sparse-gpt -v
python evaluation/evaluation.py robustness qwen --size 7b --modification pruning --pruning_amount 0.7 --pruning_method wanda -v
python evaluation/evaluation.py robustness qwen --size 7b --modification pruning --pruning_amount 0.7 --pruning_method sparse-gpt -v
python evaluation/evaluation.py robustness qwen2 --size 8b --modification pruning --pruning_amount 0.5 --pruning_method wanda -v
python evaluation/evaluation.py robustness qwen2 --size 8b --modification pruning --pruning_amount 0.5 --pruning_method sparse-gpt -v
python evaluation/evaluation.py robustness qwen2 --size 8b --modification pruning --pruning_amount 0.7 --pruning_method wanda -v
python evaluation/evaluation.py robustness qwen2 --size 8b --modification pruning --pruning_amount 0.7 --pruning_method sparse-gpt -v

echo "########## gemma ##########"
python evaluation/evaluation.py robustness gemma --size 7b --modification pruning --pruning_amount 0.5 --pruning_method wanda -v
python evaluation/evaluation.py robustness gemma --size 7b --modification pruning --pruning_amount 0.5 --pruning_method sparse-gpt -v
python evaluation/evaluation.py robustness gemma --size 7b --modification pruning --pruning_amount 0.7 --pruning_method wanda -v
python evaluation/evaluation.py robustness gemma --size 7b --modification pruning --pruning_amount 0.7 --pruning_method sparse-gpt -v

echo "########## mistral ##########"
python evaluation/evaluation.py robustness mistral --size 8b --modification pruning --pruning_amount 0.5 --pruning_method wanda -v
python evaluation/evaluation.py robustness mistral --size 8b --modification pruning --pruning_amount 0.5 --pruning_method sparse-gpt -v
python evaluation/evaluation.py robustness mistral --size 8b --modification pruning --pruning_amount 0.7 --pruning_method wanda -v
python evaluation/evaluation.py robustness mistral --size 8b --modification pruning --pruning_amount 0.7 --pruning_method sparse-gpt -v

echo "########## 3B and 4B ##########"
python evaluation/evaluation.py robustness llama --size 3b --modification pruning --pruning_amount 0.5 --pruning_method wanda -v
python evaluation/evaluation.py robustness llama --size 3b --modification pruning --pruning_amount 0.5 --pruning_method sparse-gpt -v
python evaluation/evaluation.py robustness llama --size 3b --modification pruning --pruning_amount 0.7 --pruning_method wanda -v
python evaluation/evaluation.py robustness llama --size 3b --modification pruning --pruning_amount 0.7 --pruning_method sparse-gpt -v

python evaluation/evaluation.py robustness qwen --size 4b --modification pruning --pruning_amount 0.5 --pruning_method wanda -v
python evaluation/evaluation.py robustness qwen --size 4b --modification pruning --pruning_amount 0.5 --pruning_method sparse-gpt -v
python evaluation/evaluation.py robustness qwen --size 4b --modification pruning --pruning_amount 0.7 --pruning_method wanda -v
python evaluation/evaluation.py robustness qwen --size 4b --modification pruning --pruning_amount 0.7 --pruning_method sparse-gpt -v

echo "FINISHED"

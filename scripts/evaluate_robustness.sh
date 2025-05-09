#!/usr/bin/bash

#python -u evaluation/evaluation.py robustness llama --size 3b --modification quantization --quant_bits 2 -v
#python -u evaluation/evaluation.py robustness llama --size 3b --modification quantization --quant_bits 3 -v
#python -u evaluation/evaluation.py robustness llama --size 3b --modification quantization --quant_bits 4 -v
#python -u evaluation/evaluation.py robustness llama --size 3b --modification quantization --quant_bits 8 -v
#
#python -u evaluation/evaluation.py robustness llama --size 7b --modification quantization --quant_bits 2 -v
#python -u evaluation/evaluation.py robustness llama --size 7b --modification quantization --quant_bits 3 -v
#python -u evaluation/evaluation.py robustness llama --size 7b --modification quantization --quant_bits 4 -v
#python -u evaluation/evaluation.py robustness llama --size 7b --modification quantization --quant_bits 8 -v
#
#python -u evaluation/evaluation.py robustness llama --size 8b --modification quantization --quant_bits 2 -v
#python -u evaluation/evaluation.py robustness llama --size 8b --modification quantization --quant_bits 3 -v
#python -u evaluation/evaluation.py robustness llama --size 8b --modification quantization --quant_bits 4 -v
#python -u evaluation/evaluation.py robustness llama --size 8b --modification quantization --quant_bits 8 -v

python -u evaluation/evaluation.py robustness qwen --size 4b --modification quantization --quant_bits 2 -v
python -u evaluation/evaluation.py robustness qwen --size 4b --modification quantization --quant_bits 3 -v
python -u evaluation/evaluation.py robustness qwen --size 4b --modification quantization --quant_bits 4 -v
python -u evaluation/evaluation.py robustness qwen --size 4b --modification quantization --quant_bits 8 -v

#python -u evaluation/evaluation.py robustness qwen --size 7b --modification quantization --quant_bits 2 -v
#python -u evaluation/evaluation.py robustness qwen --size 7b --modification quantization --quant_bits 3 -v
#python -u evaluation/evaluation.py robustness qwen --size 7b --modification quantization --quant_bits 4 -v
#python -u evaluation/evaluation.py robustness qwen --size 7b --modification quantization --quant_bits 8 -v

python -u evaluation/evaluation.py robustness qwen2 --size 7b --modification quantization --quant_bits 2 -v
python -u evaluation/evaluation.py robustness qwen2 --size 7b --modification quantization --quant_bits 3 -v
python -u evaluation/evaluation.py robustness qwen2 --size 7b --modification quantization --quant_bits 4 -v
python -u evaluation/evaluation.py robustness qwen2 --size 7b --modification quantization --quant_bits 8 -v

python -u evaluation/evaluation.py robustness gemma --size 7b --modification quantization --quant_bits 2 -v
python -u evaluation/evaluation.py robustness gemma --size 7b --modification quantization --quant_bits 3 -v
python -u evaluation/evaluation.py robustness gemma --size 7b --modification quantization --quant_bits 4 -v
python -u evaluation/evaluation.py robustness gemma --size 7b --modification quantization --quant_bits 8 -v

python -u evaluation/evaluation.py robustness mistral --size 8b --modification quantization --quant_bits 2 -v
python -u evaluation/evaluation.py robustness mistral --size 8b --modification quantization --quant_bits 3 -v
python -u evaluation/evaluation.py robustness mistral --size 8b --modification quantization --quant_bits 4 -v
python -u evaluation/evaluation.py robustness mistral --size 8b --modification quantization --quant_bits 8 -v

##13b llm
#python -u evaluation/evaluation.py robustness stablelm --size 12b --modification quantization --quant_bits 2 -v
#python -u evaluation/evaluation.py robustness stablelm --size 12b --modification quantization --quant_bits 3 -v
#python -u evaluation/evaluation.py robustness stablelm --size 12b --modification quantization --quant_bits 4 -v
#python -u evaluation/evaluation.py robustness stablelm --size 12b --modification quantization --quant_bits 8 -v
#
#python -u evaluation/evaluation.py robustness llama --size 13b --modification quantization --quant_bits 2 -v
#python -u evaluation/evaluation.py robustness llama --size 13b --modification quantization --quant_bits 3 -v
#python -u evaluation/evaluation.py robustness llama --size 13b --modification quantization --quant_bits 4 -v
#python -u evaluation/evaluation.py robustness llama --size 13b --modification quantization --quant_bits 8 -v
#
#python -u evaluation/evaluation.py robustness qwen --size 14b --modification quantization --quant_bits 2 -v
#python -u evaluation/evaluation.py robustness qwen --size 14b --modification quantization --quant_bits 3 -v
#python -u evaluation/evaluation.py robustness qwen --size 14b --modification quantization --quant_bits 4 -v
#python -u evaluation/evaluation.py robustness qwen --size 14b --modification quantization --quant_bits 8 -v
#
#python -u evaluation/evaluation.py robustness qwen2 --size 14b --modification quantization --quant_bits 2 -v
#python -u evaluation/evaluation.py robustness qwen2 --size 14b --modification quantization --quant_bits 3 -v
#python -u evaluation/evaluation.py robustness qwen2 --size 14b --modification quantization --quant_bits 4 -v
#python -u evaluation/evaluation.py robustness qwen2 --size 14b --modification quantization --quant_bits 8 -v

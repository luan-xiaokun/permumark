#!/bin/bash

# quantization evaluation
echo "########## llama ##########"
python evaluation/evaluation.py robustness llama --size 7b --modification quantization --quant_bits 2 -v
python evaluation/evaluation.py robustness llama --size 7b --modification quantization --quant_bits 3 -v
python evaluation/evaluation.py robustness llama --size 7b --modification quantization --quant_bits 4 -v
python evaluation/evaluation.py robustness llama --size 7b --modification quantization --quant_bits 8 -v
python evaluation/evaluation.py robustness llama --size 8b --modification quantization --quant_bits 2 -v
python evaluation/evaluation.py robustness llama --size 8b --modification quantization --quant_bits 3 -v
python evaluation/evaluation.py robustness llama --size 8b --modification quantization --quant_bits 4 -v
python evaluation/evaluation.py robustness llama --size 8b --modification quantization --quant_bits 8 -v

echo "########## qwen ##########"
python evaluation/evaluation.py robustness qwen --size 7b --modification quantization --quant_bits 2 -v
python evaluation/evaluation.py robustness qwen --size 7b --modification quantization --quant_bits 3 -v
python evaluation/evaluation.py robustness qwen --size 7b --modification quantization --quant_bits 4 -v
python evaluation/evaluation.py robustness qwen --size 7b --modification quantization --quant_bits 8 -v
python evaluation/evaluation.py robustness qwen2 --size 7b --modification quantization --quant_bits 2 -v
python evaluation/evaluation.py robustness qwen2 --size 7b --modification quantization --quant_bits 3 -v
python evaluation/evaluation.py robustness qwen2 --size 7b --modification quantization --quant_bits 4 -v
python evaluation/evaluation.py robustness qwen2 --size 7b --modification quantization --quant_bits 8 -v

echo "########## gemma ##########"
python evaluation/evaluation.py robustness gemma --size 7b --modification quantization --quant_bits 2 -v
python evaluation/evaluation.py robustness gemma --size 7b --modification quantization --quant_bits 3 -v
python evaluation/evaluation.py robustness gemma --size 7b --modification quantization --quant_bits 4 -v
python evaluation/evaluation.py robustness gemma --size 7b --modification quantization --quant_bits 8 -v

echo "########## mistral ##########"
python evaluation/evaluation.py robustness mistral --size 8b --modification quantization --quant_bits 2 -v
python evaluation/evaluation.py robustness mistral --size 8b --modification quantization --quant_bits 3 -v
python evaluation/evaluation.py robustness mistral --size 8b --modification quantization --quant_bits 4 -v
python evaluation/evaluation.py robustness mistral --size 8b --modification quantization --quant_bits 8 -v

echo "########## 3B and 4B ##########"
python evaluation/evaluation.py robustness llama --size 3b --modification quantization --quant_bits 2 -v
python evaluation/evaluation.py robustness llama --size 3b --modification quantization --quant_bits 3 -v
python evaluation/evaluation.py robustness llama --size 3b --modification quantization --quant_bits 4 -v
python evaluation/evaluation.py robustness llama --size 3b --modification quantization --quant_bits 8 -v
python evaluation/evaluation.py robustness qwen --size 4b --modification quantization --quant_bits 2 -v
python evaluation/evaluation.py robustness qwen --size 4b --modification quantization --quant_bits 3 -v
python evaluation/evaluation.py robustness qwen --size 4b --modification quantization --quant_bits 4 -v
python evaluation/evaluation.py robustness qwen --size 4b --modification quantization --quant_bits 8 -v

echo "FINISHED"

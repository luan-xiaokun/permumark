#!/bin/bash

# utility evaluation
echo "########## llama ##########"
python evaluation/evaluation.py utility llama --size 7b--batch_size 16 -v
python evaluation/evaluation.py utility llama --size 8b--batch_size 16 -v

echo "########## qwen ##########"
python evaluation/evaluation.py utility qwen --size 7b--batch_size 16 -v
python evaluation/evaluation.py utility qwen2 --size 7b--batch_size 16 -v

echo "########## gemma ##########"
python evaluation/evaluation.py utility gemma --size 7b--batch_size 8 -v

echo "########## mistral ##########"
python evaluation/evaluation.py utility mistral --size 7b--batch_size 16 -v

echo "########## 3B and 4B ##########"
python evaluation/evaluation.py utility llama --size 3b--batch_size 32 -v
python evaluation/evaluation.py utility qwen --size 4b--batch_size 32 -v

echo "########## 12B 13B 14B ##########"
python evaluation/evaluation.py utility stablelm --size 12b--batch_size 4 -v
python evaluation/evaluation.py utility llama --size 13b--batch_size 4 -v
python evaluation/evaluation.py utility qwen --size 14b--batch_size 4 -v
python evaluation/evaluation.py utility qwen2 --size 14b--batch_size 4 -v

echo "FINISHED"

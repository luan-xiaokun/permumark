#!/bin/bash

# efficiency evaluation
echo "########## llama ##########"
python evaluation/evaluation.py efficiency llama --size 7b --repeat 10 -v
python evaluation/evaluation.py efficiency llama --size 8b --repeat 10 -v

echo "########## qwen ##########"
python evaluation/evaluation.py efficiency qwen --size 7b --repeat 10 -v
python evaluation/evaluation.py efficiency qwen2 --size 7b --repeat 10 -v

echo "########## gemma ##########"
python evaluation/evaluation.py efficiency gemma --size 7b --repeat 10 -v

echo "########## mistral ##########"
python evaluation/evaluation.py efficiency mistral --size 7b --repeat 10 -v

echo "########## 3B and 4B ##########"
python evaluation/evaluation.py efficiency llama --size 3b --repeat 10 -v
python evaluation/evaluation.py efficiency qwen --size 4b --repeat 10 -v

echo "########## 12B 13B 14B ##########"
python evaluation/evaluation.py efficiency stablelm --size 12b --repeat 10 -v
python evaluation/evaluation.py efficiency llama --size 13b --repeat 10 -v
python evaluation/evaluation.py efficiency qwen --size 14b --repeat 10 -v
python evaluation/evaluation.py efficiency qwen2 --size 14b --repeat 10 -v

echo "FINISHED"

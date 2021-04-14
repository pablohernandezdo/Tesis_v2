#!/bin/bash

mkdir -p ../Results

echo "Evaluating model Cnn1_3k_10_1e4_256"
python ../eval_francia.py \
              --classifier Cnn1_3k_10 \
              --model_folder separated \
              --model_name Cnn1_3k_10_1e4_256_40 > ../Results/Cnn1_3k_10_1e4_256_40_francia.txt

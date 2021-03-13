#!/bin/bash

mkdir -p ../Results/Testing/Logs/DAS

das="DAS_dataset.hdf5"

echo "Evaluating model Cnn1_3k_10_1e4_256"
python ../eval_das.py \
              --das_dataset_path $das \
              --classifier Cnn1_3k_10 \
              --model_folder separated \
              --model_name Cnn1_3k_10_1e4_256_40 > ../Results/Testing/Logs/DAS/Cnn1_3k_10_1e4_256_40.txt

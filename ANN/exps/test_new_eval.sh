#!/bin/bash

mkdir -p ../Analysis/logs/train/step4
mkdir -p ../Analysis/logs/eval/step4
mkdir -p ../models/step4

tst="Test_data.hdf5"
trn="Train_data.hdf5"

echo "Evaluating model 2h5h5k_1e3_256"
python ../eval.py \
              --test_path $tst \
              --train_path $trn \
              --model_folder step4 \
              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_20 > ../Analysis/logs/eval/step4/2h5h5k_1e3_256_20.txt

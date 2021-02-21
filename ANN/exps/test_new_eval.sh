#!/bin/bash

mkdir -p ../Analysis/logs/train/step4
mkdir -p ../Analysis/logs/eval/step4
mkdir -p ../models/step4

tst="Test_data.hdf5"
trn="Train_data.hdf5"

echo "Evaluating model 2h5h4k_5e5_256_35"
python ../eval.py \
              --test_path $tst \
              --train_path $trn \
              --model_folder step4 \
              --classifier 2h5h4k --model_name 2h5h4k_5e5_256_35 > ../Analysis/logs/eval/step4/2h5h4k_5e5_256_35.txt

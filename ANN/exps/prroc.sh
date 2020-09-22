#!/bin/bash

mkdir -p ../logs/eval_curves

tst="Test_data.hdf5"
trn="Train_data.hdf5"

echo "Evaluating model 1h6k_1e4_256"
python ../eval_curves.py --train_path $trn --test_path $tst \
              --classifier 1h6k --model_name 1h6k_1e4_256 > ../logs/eval_curves/1h6k_1e4_256.txt

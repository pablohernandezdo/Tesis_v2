#!/bin/bash

mkdir -p ../logs/eval_curves

tst="Test_data.hdf5"
trn="Train_data.hdf5"

echo "Evaluating model 2h1h5k_1e4_256"
python ../eval_curves.py --train_path $trn --test_path $tst \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256 > ../logs/eval_curves/2h1h5k_1e4_256.txt

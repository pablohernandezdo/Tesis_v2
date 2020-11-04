#!/bin/bash

mkdir -p ../logs/train
mkdir -p ../logs/eval
mkdir -p ../models

tst="Test_data.hdf5"

echo "Evaluating model 2c3k3k_1e3_256"
python ../eval_curves.py --test_path $tst \
              --classifier 2c3k3k --model_name 2c3k3k_1e3_256 > ../logs/eval/2c3k3k_1e3_256.txt

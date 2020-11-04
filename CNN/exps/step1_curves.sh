#!/bin/bash

mkdir -p ../logs/train
mkdir -p ../logs/eval
mkdir -p ../models

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

echo "Evaluating model 1c3k_1e3_256"
python ../eval_curves.py --test_path $tst \
              --classifier 1c3k --model_name 1c3k_1e3_256 > ../logs/eval/1c3k_1e3_256.txt

echo "Evaluating model 1c3k_1e4_256"
python ../eval_curves.py --test_path $tst \
              --classifier 1c3k --model_name 1c3k_1e4_256 > ../logs/eval/1c3k_1e4_256.txt

echo "Evaluating model 1c3k_1e5_256"
python ../eval_curves.py --test_path $tst \
              --classifier 1c3k --model_name 1c3k_1e5_256 > ../logs/eval/1c3k_1e5_256.txt

echo "Evaluating model 1c3k_1e6_256"
python ../eval_curves.py --test_path $tst \
              --classifier 1c3k --model_name 1c3k_1e6_256 > ../logs/eval/1c3k_1e6_256.txt

echo "Evaluating model 1c3k3k_1e3_256"
python ../eval_curves.py --test_path $tst \
              --classifier 1c3k3k --model_name 1c3k3k_1e3_256 > ../logs/eval/1c3k3k_1e3_256.txt

echo "Evaluating model 1c3k3k_1e4_256"
python ../eval_curves.py --test_path $tst \
              --classifier 1c3k3k --model_name 1c3k3k_1e4_256 > ../logs/eval/1c3k3k_1e4_256.txt

echo "Evaluating model 1c3k3k_1e5_256"
python ../eval_curves.py --test_path $tst \
              --classifier 1c3k3k --model_name 1c3k3k_1e5_256 > ../logs/eval/1c3k3k_1e5_256.txt

echo "Evaluating model 1c3k3k_1e6_256"
python ../eval_curves.py --test_path $tst \
              --classifier 1c3k3k --model_name 1c3k3k_1e6_256 > ../logs/eval/1c3k3k_1e6_256.txt

echo "Evaluating model 2c3k_1e3_256"
python ../eval_curves.py --test_path $tst \
              --classifier 2c3k --model_name 2c3k_1e3_256 > ../logs/eval/2c3k_1e3_256.txt

echo "Evaluating model 2c3k_1e4_256"
python ../eval_curves.py --test_path $tst \
              --classifier 2c3k --model_name 2c3k_1e4_256 > ../logs/eval/2c3k_1e4_256.txt

echo "Evaluating model 2c3k_1e5_256"
python ../eval_curves.py --test_path $tst \
              --classifier 2c3k --model_name 2c3k_1e5_256 > ../logs/eval/2c3k_1e5_256.txt

echo "Evaluating model 2c3k_1e6_256"
python ../eval_curves.py --test_path $tst \
              --classifier 2c3k --model_name 2c3k_1e6_256 > ../logs/eval/2c3k_1e6_256.txt

echo "Evaluating model 2c3k3k_1e3_256"
python ../eval_curves.py --test_path $tst \
              --classifier 2c3k3k --model_name 2c3k3k_1e3_256 > ../logs/eval/2c3k3k_1e3_256.txt

echo "Evaluating model 2c3k3k_1e4_256"
python ../eval_curves.py --test_path $tst \
              --classifier 2c3k3k --model_name 2c3k3k_1e4_256 > ../logs/eval/2c3k3k_1e4_256.txt

echo "Evaluating model 2c3k3k_1e5_256"
python ../eval_curves.py --test_path $tst \
              --classifier 2c3k3k --model_name 2c3k3k_1e5_256 > ../logs/eval/2c3k3k_1e5_256.txt

echo "Evaluating model 2c3k3k_1e6_256"
python ../eval_curves.py $trn --test_path $tst \
              --classifier 2c3k3k --model_name 2c3k3k_1e6_256 > ../logs/eval/2c3k3k_1e6_256.txt

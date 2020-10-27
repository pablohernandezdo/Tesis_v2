#!/bin/bash

mkdir -p ../logs/eval_curves

tst="Test_data.hdf5"
trn="Train_data.hdf5"

echo "Evaluating model 2h1k5k_1e3_256_30"
python ../eval_curves.py --batch_size 256 --test_path $tst \
              --classifier 2h1k5k --model_name 2h1k5k_1e3_256_30 > ../logs/eval_curves/2h1k5k_1e3_256_30.txt

echo "Evaluating model 2h5h6k_1e4_256_40"
python ../eval_curves.py --batch_size 256 --test_path $tst \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_40 > ../logs/eval_curves/2h5h6k_1e4_256_40.txt

echo "Evaluating model 2h5h6k_1e4_256_35"
python ../eval_curves.py --batch_size 256 --test_path $tst \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_35 > ../logs/eval_curves/2h5h6k_1e4_256_35.txt

echo "Evaluating model 2h5h5k_1e4_256_25"
python ../eval_curves.py --batch_size 256 --test_path $tst \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_25 > ../logs/eval_curves/2h5h5k_1e4_256_25.txt

echo "Evaluating model 2h5h6k_1e4_256_30"
python ../eval_curves.py --batch_size 256 --test_path $tst \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_30 > ../logs/eval_curves/2h5h6k_1e4_256_30.txt

echo "Evaluating model 2h5h5k_1e4_256_35"
python ../eval_curves.py --batch_size 256 --test_path $tst \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_35 > ../logs/eval_curves/2h5h5k_1e4_256_35.txt

echo "Evaluating model 2h5h5k_1e4_256_30"
python ../eval_curves.py --batch_size 256 --test_path $tst \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_30 > ../logs/eval_curves/2h5h5k_1e4_256_30.txt

echo "Evaluating model 2h5h6k_5e5_256_40"
python ../eval_curves.py --batch_size 256 --test_path $tst \
              --classifier 2h5h6k --model_name 2h5h6k_5e5_256_40 > ../logs/eval_curves/2h5h6k_5e5_256_40.txt

echo "Evaluating model 2h5h5k_1e4_256_35"
python ../eval_curves.py --batch_size 256 --test_path $tst \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_35 > ../logs/eval_curves/2h5h5k_1e4_256_35.txt

echo "Evaluating model 2h5h4k_1e4_256_40"
python ../eval_curves.py --batch_size 256 --test_path $tst \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_40 > ../logs/eval_curves/2h5h4k_1e4_256_40.txt

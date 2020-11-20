#!/bin/bash

mkdir -p ../Analysis/logs/eval/avgmodels

tst="Test_data.hdf5"

#echo "Evaluating model 2h1h10_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step3 \
#              --classifier 2h1h10 --model_name 2h1h10_1e4_256_30 > ../Analysis/logs/eval/avgmodels/2h1h10_1e4_256_30.txt
#
#echo "Evaluating model 1h5h_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step3 \
#              --classifier 1h5h --model_name 1h5h_1e4_256_30 > ../Analysis/logs/eval/avgmodels/1h5h_1e4_256_30.txt
#
echo "Evaluating model 2h10_2k_1e4_256_30"
python ../eval_curves.py --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_2k --model_name 2h10_2k_1e4_256_30 > ../Analysis/logs/eval/avgmodels/2h10_2k_1e4_256_30.txt

echo "Evaluating model 2h1_2k_1e4_256_30"
python ../eval_curves.py --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_2k --model_name 2h1_2k_1e4_256_30 > ../Analysis/logs/eval/avgmodels/2h1_2k_1e4_256_30.txt
#
#echo "Evaluating model 2h5h10_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step3 \
#              --classifier 2h5h10 --model_name 2h5h10_1e4_256_30 > ../Analysis/logs/eval/avgmodels/2h5h10_1e4_256_30.txt
#
#echo "Evaluating model 2h1h1h_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step3 \
#              --classifier 2h1h1h --model_name 2h1h1h_1e4_256_30 > ../Analysis/logs/eval/avgmodels/2h1h1h_1e4_256_30.txt
#
echo "Evaluating model 2h10_1k_1e4_256_30"
python ../eval_curves.py --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_1k --model_name 2h10_1k_1e4_256_30 > ../Analysis/logs/eval/avgmodels/2h10_1k_1e4_256_30.txt

#echo "Evaluating model 1h1k_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step3 \
#              --classifier 1h1k --model_name 1h1k_1e4_256_30 > ../Analysis/logs/eval/avgmodels/1h1k_1e4_256_30.txt
#
#echo "Evaluating model 2h5h1h_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step3 \
#              --classifier 2h5h1h --model_name 2h5h1h_1e4_256_30 > ../Analysis/logs/eval/avgmodels/2h5h1h_1e4_256_30.txt
#
#echo "Evaluating model 2h1h1k_1e4_256_30"
#python ../eval_curves.py --test_path $tst \
#              --model_folder step3 \
#              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_30 > ../Analysis/logs/eval/avgmodels/2h1h1k_1e4_256_30.txt

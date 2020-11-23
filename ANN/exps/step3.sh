#!/bin/bash

mkdir -p ../Analysis/logs/train/step3
mkdir -p ../Analysis/logs/eval/step3
mkdir -p ../models/step3

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

#echo "Training model 1h6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 1h6k --model_name 1h6k_1e4_256_30 > ../Analysis/logs/train/step3/1h6k_1e4_256_30.txt
#
echo "Evaluating model 1h6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 1h6k --model_name 1h6k_1e4_256_30 > ../Analysis/logs/eval/step3/1h6k_1e4_256_30.txt
#
#echo "Training model 1h5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 1h5k --model_name 1h5k_1e4_256_30 > ../Analysis/logs/train/step3/1h5k_1e4_256_30.txt
#
echo "Evaluating model 1h5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 1h5k --model_name 1h5k_1e4_256_30 > ../Analysis/logs/eval/step3/1h5k_1e4_256_30.txt
#
#echo "Training model 1h4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 1h4k --model_name 1h4k_1e4_256_30 > ../Analysis/logs/train/step3/1h4k_1e4_256_30.txt
#
echo "Evaluating model 1h4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 1h4k --model_name 1h4k_1e4_256_30 > ../Analysis/logs/eval/step3/1h4k_1e4_256_30.txt
#
#echo "Training model 1h3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 1h3k --model_name 1h3k_1e4_256_30 > ../Analysis/logs/train/step3/1h3k_1e4_256_30.txt
#
echo "Evaluating model 1h3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 1h3k --model_name 1h3k_1e4_256_30 > ../Analysis/logs/eval/step3/1h3k_1e4_256_30.txt
#
#echo "Training model 1h2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 1h2k --model_name 1h2k_1e4_256_30 > ../Analysis/logs/train/step3/1h2k_1e4_256_30.txt
#
echo "Evaluating model 1h2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 1h2k --model_name 1h2k_1e4_256_30 > ../Analysis/logs/eval/step3/1h2k_1e4_256_30.txt
#
#echo "Training model 1h1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 1h1k --model_name 1h1k_1e4_256_30 > ../Analysis/logs/train/step3/1h1k_1e4_256_30.txt
#
echo "Evaluating model 1h1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 1h1k --model_name 1h1k_1e4_256_30 > ../Analysis/logs/eval/step3/1h1k_1e4_256_30.txt
#
#echo "Training model 1h5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 1h5h --model_name 1h5h_1e4_256_30 > ../Analysis/logs/train/step3/1h5h_1e4_256_30.txt
#
echo "Evaluating model 1h5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 1h5h --model_name 1h5h_1e4_256_30 > ../Analysis/logs/eval/step3/1h5h_1e4_256_30.txt
#
#echo "Training model 1h1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 1h1h --model_name 1h1h_1e4_256_30 > ../Analysis/logs/train/step3/1h1h_1e4_256_30.txt
#
echo "Evaluating model 1h1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 1h1h --model_name 1h1h_1e4_256_30 > ../Analysis/logs/eval/step3/1h1h_1e4_256_30.txt
#
#echo "Training model 1h10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 1h10 --model_name 1h10_1e4_256_30 > ../Analysis/logs/train/step3/1h10_1e4_256_30.txt
#
echo "Evaluating model 1h10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 1h10 --model_name 1h10_1e4_256_30 > ../Analysis/logs/eval/step3/1h10_1e4_256_30.txt
#
#echo "Training model 1h1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 1h1 --model_name 1h1_1e4_256_30 > ../Analysis/logs/train/step3/1h1_1e4_256_30.txt
#
echo "Evaluating model 1h1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 1h1 --model_name 1h1_1e4_256_30 > ../Analysis/logs/eval/step3/1h1_1e4_256_30.txt
#
#echo "Training model 2h6k6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h6k6k --model_name 2h6k6k_1e4_256_30 > ../Analysis/logs/train/step3/2h6k6k_1e4_256_30.txt
#
echo "Evaluating model 2h6k6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h6k6k --model_name 2h6k6k_1e4_256_30 > ../Analysis/logs/eval/step3/2h6k6k_1e4_256_30.txt
#
#
#echo "Training model 2h6k5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h6k5k --model_name 2h6k5k_1e4_256_30 > ../Analysis/logs/train/step3/2h6k5k_1e4_256_30.txt
#
echo "Evaluating model 2h6k5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h6k5k --model_name 2h6k5k_1e4_256_30 > ../Analysis/logs/eval/step3/2h6k5k_1e4_256_30.txt
#
#echo "Training model 2h6k4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h6k4k --model_name 2h6k4k_1e4_256_30 > ../Analysis/logs/train/step3/2h6k4k_1e4_256_30.txt
#
echo "Evaluating model 2h6k4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h6k4k --model_name 2h6k4k_1e4_256_30 > ../Analysis/logs/eval/step3/2h6k4k_1e4_256_30.txt
#
#echo "Training model 2h6k3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h6k3k --model_name 2h6k3k_1e4_256_30 > ../Analysis/logs/train/step3/2h6k3k_1e4_256_30.txt
#
echo "Evaluating model 2h6k3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h6k3k --model_name 2h6k3k_1e4_256_30 > ../Analysis/logs/eval/step3/2h6k3k_1e4_256_30.txt
#
#echo "Training model 2h6k2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h6k2k --model_name 2h6k2k_1e4_256_30 > ../Analysis/logs/train/step3/2h6k2k_1e4_256_30.txt
#
echo "Evaluating model 2h6k2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h6k2k --model_name 2h6k2k_1e4_256_30 > ../Analysis/logs/eval/step3/2h6k2k_1e4_256_30.txt
#
#echo "Training model 2h6k1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h6k1k --model_name 2h6k1k_1e4_256_30 > ../Analysis/logs/train/step3/2h6k1k_1e4_256_30.txt
#
echo "Evaluating model 2h6k1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h6k1k --model_name 2h6k1k_1e4_256_30 > ../Analysis/logs/eval/step3/2h6k1k_1e4_256_30.txt
#
#echo "Training model 2h6k5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h6k5h --model_name 2h6k5h_1e4_256_30 > ../Analysis/logs/train/step3/2h6k5h_1e4_256_30.txt
#
echo "Evaluating model 2h6k5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h6k5h --model_name 2h6k5h_1e4_256_30 > ../Analysis/logs/eval/step3/2h6k5h_1e4_256_30.txt
#
#echo "Training model 2h6k1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h6k1h --model_name 2h6k1h_1e4_256_30 > ../Analysis/logs/train/step3/2h6k1h_1e4_256_30.txt
#
echo "Evaluating model 2h6k1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h6k1h --model_name 2h6k1h_1e4_256_30 > ../Analysis/logs/eval/step3/2h6k1h_1e4_256_30.txt
#
#echo "Training model 2h6k10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h6k10 --model_name 2h6k10_1e4_256_30 > ../Analysis/logs/train/step3/2h6k10_1e4_256_30.txt
#
echo "Evaluating model 2h6k10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h6k10 --model_name 2h6k10_1e4_256_30 > ../Analysis/logs/eval/step3/2h6k10_1e4_256_30.txt
#
#echo "Training model 2h6k1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h6k1 --model_name 2h6k1_1e4_256_30 > ../Analysis/logs/train/step3/2h6k1_1e4_256_30.txt
#
echo "Evaluating model 2h6k1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h6k1 --model_name 2h6k1_1e4_256_30 > ../Analysis/logs/eval/step3/2h6k1_1e4_256_30.txt
#
#echo "Training model 2h5k6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_30 > ../Analysis/logs/train/step3/2h5k6k_1e4_256_30.txt
#
echo "Evaluating model 2h5k6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5k6k --model_name 2h5k6k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5k6k_1e4_256_30.txt
#
#
#echo "Training model 2h5k5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_30 > ../Analysis/logs/train/step3/2h5k5k_1e4_256_30.txt
#
echo "Evaluating model 2h5k5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5k5k --model_name 2h5k5k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5k5k_1e4_256_30.txt
#
#echo "Training model 2h5k4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_30 > ../Analysis/logs/train/step3/2h5k4k_1e4_256_30.txt
#
echo "Evaluating model 2h5k4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5k4k --model_name 2h5k4k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5k4k_1e4_256_30.txt
#
#echo "Training model 2h5k3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_30 > ../Analysis/logs/train/step3/2h5k3k_1e4_256_30.txt
#
echo "Evaluating model 2h5k3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5k3k --model_name 2h5k3k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5k3k_1e4_256_30.txt
#
#echo "Training model 2h5k2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5k2k --model_name 2h5k2k_1e4_256_30 > ../Analysis/logs/train/step3/2h5k2k_1e4_256_30.txt
#
echo "Evaluating model 2h5k2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5k2k --model_name 2h5k2k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5k2k_1e4_256_30.txt
#
#echo "Training model 2h5k1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5k1k --model_name 2h5k1k_1e4_256_30 > ../Analysis/logs/train/step3/2h5k1k_1e4_256_30.txt
#
echo "Evaluating model 2h5k1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5k1k --model_name 2h5k1k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5k1k_1e4_256_30.txt
#
#echo "Training model 2h5k5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_30 > ../Analysis/logs/train/step3/2h5k5h_1e4_256_30.txt
#
echo "Evaluating model 2h5k5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5k5h --model_name 2h5k5h_1e4_256_30 > ../Analysis/logs/eval/step3/2h5k5h_1e4_256_30.txt
#
#echo "Training model 2h5k1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5k1h --model_name 2h5k1h_1e4_256_30 > ../Analysis/logs/train/step3/2h5k1h_1e4_256_30.txt
#
echo "Evaluating model 2h5k1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5k1h --model_name 2h5k1h_1e4_256_30 > ../Analysis/logs/eval/step3/2h5k1h_1e4_256_30.txt
#
#echo "Training model 2h5k10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5k10 --model_name 2h5k10_1e4_256_30 > ../Analysis/logs/train/step3/2h5k10_1e4_256_30.txt
#
echo "Evaluating model 2h5k10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5k10 --model_name 2h5k10_1e4_256_30 > ../Analysis/logs/eval/step3/2h5k10_1e4_256_30.txt
#
#echo "Training model 2h5k1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5k1 --model_name 2h5k1_1e4_256_30 > ../Analysis/logs/train/step3/2h5k1_1e4_256_30.txt
#
echo "Evaluating model 2h5k1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5k1 --model_name 2h5k1_1e4_256_30 > ../Analysis/logs/eval/step3/2h5k1_1e4_256_30.txt
#
#echo "Training model 2h4k6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_30 > ../Analysis/logs/train/step3/2h4k6k_1e4_256_30.txt
#
echo "Evaluating model 2h4k6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h4k6k --model_name 2h4k6k_1e4_256_30 > ../Analysis/logs/eval/step3/2h4k6k_1e4_256_30.txt
#
#
#echo "Training model 2h4k5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_30 > ../Analysis/logs/train/step3/2h4k5k_1e4_256_30.txt
#
echo "Evaluating model 2h4k5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h4k5k --model_name 2h4k5k_1e4_256_30 > ../Analysis/logs/eval/step3/2h4k5k_1e4_256_30.txt
#
#echo "Training model 2h4k4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h4k4k --model_name 2h4k4k_1e4_256_30 > ../Analysis/logs/train/step3/2h4k4k_1e4_256_30.txt
#
echo "Evaluating model 2h4k4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h4k4k --model_name 2h4k4k_1e4_256_30 > ../Analysis/logs/eval/step3/2h4k4k_1e4_256_30.txt
#
#echo "Training model 2h4k3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_30 > ../Analysis/logs/train/step3/2h4k3k_1e4_256_30.txt
#
echo "Evaluating model 2h4k3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h4k3k --model_name 2h4k3k_1e4_256_30 > ../Analysis/logs/eval/step3/2h4k3k_1e4_256_30.txt
#
#echo "Training model 2h4k2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h4k2k --model_name 2h4k2k_1e4_256_30 > ../Analysis/logs/train/step3/2h4k2k_1e4_256_30.txt
#
echo "Evaluating model 2h4k2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h4k2k --model_name 2h4k2k_1e4_256_30 > ../Analysis/logs/eval/step3/2h4k2k_1e4_256_30.txt
#
#echo "Training model 2h4k1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h4k1k --model_name 2h4k1k_1e4_256_30 > ../Analysis/logs/train/step3/2h4k1k_1e4_256_30.txt
#
echo "Evaluating model 2h4k1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h4k1k --model_name 2h4k1k_1e4_256_30 > ../Analysis/logs/eval/step3/2h4k1k_1e4_256_30.txt
#
#echo "Training model 2h4k5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h4k5h --model_name 2h4k5h_1e4_256_30 > ../Analysis/logs/train/step3/2h4k5h_1e4_256_30.txt
#
echo "Evaluating model 2h4k5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h4k5h --model_name 2h4k5h_1e4_256_30 > ../Analysis/logs/eval/step3/2h4k5h_1e4_256_30.txt
#
#echo "Training model 2h4k1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h4k1h --model_name 2h4k1h_1e4_256_30 > ../Analysis/logs/train/step3/2h4k1h_1e4_256_30.txt
#
echo "Evaluating model 2h4k1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h4k1h --model_name 2h4k1h_1e4_256_30 > ../Analysis/logs/eval/step3/2h4k1h_1e4_256_30.txt
#
#echo "Training model 2h4k10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h4k10 --model_name 2h4k10_1e4_256_30 > ../Analysis/logs/train/step3/2h4k10_1e4_256_30.txt
#
echo "Evaluating model 2h4k10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h4k10 --model_name 2h4k10_1e4_256_30 > ../Analysis/logs/eval/step3/2h4k10_1e4_256_30.txt
#
#echo "Training model 2h4k1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h4k1 --model_name 2h4k1_1e4_256_30 > ../Analysis/logs/train/step3/2h4k1_1e4_256_30.txt
#
echo "Evaluating model 2h4k1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h4k1 --model_name 2h4k1_1e4_256_30 > ../Analysis/logs/eval/step3/2h4k1_1e4_256_30.txt
#
#echo "Training model 2h3k6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_30 > ../Analysis/logs/train/step3/2h3k6k_1e4_256_30.txt
#
echo "Evaluating model 2h3k6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h3k6k --model_name 2h3k6k_1e4_256_30 > ../Analysis/logs/eval/step3/2h3k6k_1e4_256_30.txt
#
#
#echo "Training model 2h3k5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_30 > ../Analysis/logs/train/step3/2h3k5k_1e4_256_30.txt
#
echo "Evaluating model 2h3k5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h3k5k --model_name 2h3k5k_1e4_256_30 > ../Analysis/logs/eval/step3/2h3k5k_1e4_256_30.txt
#
#echo "Training model 2h3k4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h3k4k --model_name 2h3k4k_1e4_256_30 > ../Analysis/logs/train/step3/2h3k4k_1e4_256_30.txt
#
echo "Evaluating model 2h3k4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h3k4k --model_name 2h3k4k_1e4_256_30 > ../Analysis/logs/eval/step3/2h3k4k_1e4_256_30.txt
#
#echo "Training model 2h3k3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h3k3k --model_name 2h3k3k_1e4_256_30 > ../Analysis/logs/train/step3/2h3k3k_1e4_256_30.txt
#
echo "Evaluating model 2h3k3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h3k3k --model_name 2h3k3k_1e4_256_30 > ../Analysis/logs/eval/step3/2h3k3k_1e4_256_30.txt
#
#echo "Training model 2h3k2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h3k2k --model_name 2h3k2k_1e4_256_30 > ../Analysis/logs/train/step3/2h3k2k_1e4_256_30.txt
#
echo "Evaluating model 2h3k2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h3k2k --model_name 2h3k2k_1e4_256_30 > ../Analysis/logs/eval/step3/2h3k2k_1e4_256_30.txt
#
#echo "Training model 2h3k1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h3k1k --model_name 2h3k1k_1e4_256_30 > ../Analysis/logs/train/step3/2h3k1k_1e4_256_30.txt
#
echo "Evaluating model 2h3k1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h3k1k --model_name 2h3k1k_1e4_256_30 > ../Analysis/logs/eval/step3/2h3k1k_1e4_256_30.txt
#
#echo "Training model 2h3k5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h3k5h --model_name 2h3k5h_1e4_256_30 > ../Analysis/logs/train/step3/2h3k5h_1e4_256_30.txt
#
echo "Evaluating model 2h3k5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h3k5h --model_name 2h3k5h_1e4_256_30 > ../Analysis/logs/eval/step3/2h3k5h_1e4_256_30.txt
#
#echo "Training model 2h3k1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h3k1h --model_name 2h3k1h_1e4_256_30 > ../Analysis/logs/train/step3/2h3k1h_1e4_256_30.txt
#
echo "Evaluating model 2h3k1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h3k1h --model_name 2h3k1h_1e4_256_30 > ../Analysis/logs/eval/step3/2h3k1h_1e4_256_30.txt
#
#echo "Training model 2h3k10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h3k10 --model_name 2h3k10_1e4_256_30 > ../Analysis/logs/train/step3/2h3k10_1e4_256_30.txt
#
echo "Evaluating model 2h3k10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h3k10 --model_name 2h3k10_1e4_256_30 > ../Analysis/logs/eval/step3/2h3k10_1e4_256_30.txt
#
#echo "Training model 2h3k1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h3k1 --model_name 2h3k1_1e4_256_30 > ../Analysis/logs/train/step3/2h3k1_1e4_256_30.txt
#
echo "Evaluating model 2h3k1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h3k1 --model_name 2h3k1_1e4_256_30 > ../Analysis/logs/eval/step3/2h3k1_1e4_256_30.txt
#
#echo "Training model 2h2k6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_30 > ../Analysis/logs/train/step3/2h2k6k_1e4_256_30.txt
#
echo "Evaluating model 2h2k6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h2k6k --model_name 2h2k6k_1e4_256_30 > ../Analysis/logs/eval/step3/2h2k6k_1e4_256_30.txt
#
#
#echo "Training model 2h2k5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_30 > ../Analysis/logs/train/step3/2h2k5k_1e4_256_30.txt
#
echo "Evaluating model 2h2k5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h2k5k --model_name 2h2k5k_1e4_256_30 > ../Analysis/logs/eval/step3/2h2k5k_1e4_256_30.txt
#
#echo "Training model 2h2k4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_30 > ../Analysis/logs/train/step3/2h2k4k_1e4_256_30.txt
#
echo "Evaluating model 2h2k4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h2k4k --model_name 2h2k4k_1e4_256_30 > ../Analysis/logs/eval/step3/2h2k4k_1e4_256_30.txt
#
#echo "Training model 2h2k3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_30 > ../Analysis/logs/train/step3/2h2k3k_1e4_256_30.txt
#
echo "Evaluating model 2h2k3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h2k3k --model_name 2h2k3k_1e4_256_30 > ../Analysis/logs/eval/step3/2h2k3k_1e4_256_30.txt
#
#echo "Training model 2h2k2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_30 > ../Analysis/logs/train/step3/2h2k2k_1e4_256_30.txt
#
echo "Evaluating model 2h2k2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h2k2k --model_name 2h2k2k_1e4_256_30 > ../Analysis/logs/eval/step3/2h2k2k_1e4_256_30.txt
#
#echo "Training model 2h2k1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h2k1k --model_name 2h2k1k_1e4_256_30 > ../Analysis/logs/train/step3/2h2k1k_1e4_256_30.txt
#
echo "Evaluating model 2h2k1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h2k1k --model_name 2h2k1k_1e4_256_30 > ../Analysis/logs/eval/step3/2h2k1k_1e4_256_30.txt
#
#echo "Training model 2h2k5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h2k5h --model_name 2h2k5h_1e4_256_30 > ../Analysis/logs/train/step3/2h2k5h_1e4_256_30.txt
#
echo "Evaluating model 2h2k5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h2k5h --model_name 2h2k5h_1e4_256_30 > ../Analysis/logs/eval/step3/2h2k5h_1e4_256_30.txt
#
#echo "Training model 2h2k1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h2k1h --model_name 2h2k1h_1e4_256_30 > ../Analysis/logs/train/step3/2h2k1h_1e4_256_30.txt
#
echo "Evaluating model 2h2k1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h2k1h --model_name 2h2k1h_1e4_256_30 > ../Analysis/logs/eval/step3/2h2k1h_1e4_256_30.txt
#
#echo "Training model 2h2k10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h2k10 --model_name 2h2k10_1e4_256_30 > ../Analysis/logs/train/step3/2h2k10_1e4_256_30.txt
#
echo "Evaluating model 2h2k10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h2k10 --model_name 2h2k10_1e4_256_30 > ../Analysis/logs/eval/step3/2h2k10_1e4_256_30.txt
#
#echo "Training model 2h2k1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h2k1 --model_name 2h2k1_1e4_256_30 > ../Analysis/logs/train/step3/2h2k1_1e4_256_30.txt
#
echo "Evaluating model 2h2k1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h2k1 --model_name 2h2k1_1e4_256_30 > ../Analysis/logs/eval/step3/2h2k1_1e4_256_30.txt
#
#echo "Training model 2h1k6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_30 > ../Analysis/logs/train/step3/2h1k6k_1e4_256_30.txt
#
echo "Evaluating model 2h1k6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1k6k --model_name 2h1k6k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1k6k_1e4_256_30.txt
#
#
#echo "Training model 2h1k5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_30 > ../Analysis/logs/train/step3/2h1k5k_1e4_256_30.txt
#
echo "Evaluating model 2h1k5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1k5k --model_name 2h1k5k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1k5k_1e4_256_30.txt
#
#echo "Training model 2h1k4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_30 > ../Analysis/logs/train/step3/2h1k4k_1e4_256_30.txt
#
echo "Evaluating model 2h1k4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1k4k --model_name 2h1k4k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1k4k_1e4_256_30.txt
#
#echo "Training model 2h1k3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_30 > ../Analysis/logs/train/step3/2h1k3k_1e4_256_30.txt
#
echo "Evaluating model 2h1k3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1k3k --model_name 2h1k3k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1k3k_1e4_256_30.txt
#
#echo "Training model 2h1k2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_30 > ../Analysis/logs/train/step3/2h1k2k_1e4_256_30.txt
#
echo "Evaluating model 2h1k2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1k2k --model_name 2h1k2k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1k2k_1e4_256_30.txt
#
#echo "Training model 2h1k1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k1k --model_name 2h1k1k_1e4_256_30 > ../Analysis/logs/train/step3/2h1k1k_1e4_256_30.txt
#
echo "Evaluating model 2h1k1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1k1k --model_name 2h1k1k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1k1k_1e4_256_30.txt
#
#echo "Training model 2h1k5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_30 > ../Analysis/logs/train/step3/2h1k5h_1e4_256_30.txt
#
echo "Evaluating model 2h1k5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1k5h --model_name 2h1k5h_1e4_256_30 > ../Analysis/logs/eval/step3/2h1k5h_1e4_256_30.txt
#
#echo "Training model 2h1k1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k1h --model_name 2h1k1h_1e4_256_30 > ../Analysis/logs/train/step3/2h1k1h_1e4_256_30.txt
#
echo "Evaluating model 2h1k1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1k1h --model_name 2h1k1h_1e4_256_30 > ../Analysis/logs/eval/step3/2h1k1h_1e4_256_30.txt
#
#echo "Training model 2h1k10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k10 --model_name 2h1k10_1e4_256_30 > ../Analysis/logs/train/step3/2h1k10_1e4_256_30.txt
#
echo "Evaluating model 2h1k10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1k10 --model_name 2h1k10_1e4_256_30 > ../Analysis/logs/eval/step3/2h1k10_1e4_256_30.txt
#
#echo "Training model 2h1k1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1k1 --model_name 2h1k1_1e4_256_30 > ../Analysis/logs/train/step3/2h1k1_1e4_256_30.txt
#
echo "Evaluating model 2h1k1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1k1 --model_name 2h1k1_1e4_256_30 > ../Analysis/logs/eval/step3/2h1k1_1e4_256_30.txt
#
#echo "Training model 2h5h6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_30 > ../Analysis/logs/train/step3/2h5h6k_1e4_256_30.txt
#
echo "Evaluating model 2h5h6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5h6k --model_name 2h5h6k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5h6k_1e4_256_30.txt
#
#
#echo "Training model 2h5h5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_30 > ../Analysis/logs/train/step3/2h5h5k_1e4_256_30.txt
#
echo "Evaluating model 2h5h5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5h5k --model_name 2h5h5k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5h5k_1e4_256_30.txt
#
#echo "Training model 2h5h4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_30 > ../Analysis/logs/train/step3/2h5h4k_1e4_256_30.txt
#
echo "Evaluating model 2h5h4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5h4k --model_name 2h5h4k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5h4k_1e4_256_30.txt
#
#echo "Training model 2h5h3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_30 > ../Analysis/logs/train/step3/2h5h3k_1e4_256_30.txt
#
echo "Evaluating model 2h5h3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5h3k --model_name 2h5h3k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5h3k_1e4_256_30.txt
#
#echo "Training model 2h5h2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_30 > ../Analysis/logs/train/step3/2h5h2k_1e4_256_30.txt
#
echo "Evaluating model 2h5h2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5h2k --model_name 2h5h2k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5h2k_1e4_256_30.txt
#
#echo "Training model 2h5h1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_30 > ../Analysis/logs/train/step3/2h5h1k_1e4_256_30.txt
#
echo "Evaluating model 2h5h1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5h1k --model_name 2h5h1k_1e4_256_30 > ../Analysis/logs/eval/step3/2h5h1k_1e4_256_30.txt
#
#echo "Training model 2h5h5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5h5h --model_name 2h5h5h_1e4_256_30 > ../Analysis/logs/train/step3/2h5h5h_1e4_256_30.txt
#
echo "Evaluating model 2h5h5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5h5h --model_name 2h5h5h_1e4_256_30 > ../Analysis/logs/eval/step3/2h5h5h_1e4_256_30.txt
#
#echo "Training model 2h5h1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5h1h --model_name 2h5h1h_1e4_256_30 > ../Analysis/logs/train/step3/2h5h1h_1e4_256_30.txt
#
echo "Evaluating model 2h5h1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5h1h --model_name 2h5h1h_1e4_256_30 > ../Analysis/logs/eval/step3/2h5h1h_1e4_256_30.txt
#
#echo "Training model 2h5h10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5h10 --model_name 2h5h10_1e4_256_30 > ../Analysis/logs/train/step3/2h5h10_1e4_256_30.txt
#
echo "Evaluating model 2h5h10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5h10 --model_name 2h5h10_1e4_256_30 > ../Analysis/logs/eval/step3/2h5h10_1e4_256_30.txt
#
#echo "Training model 2h5h1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h5h1 --model_name 2h5h1_1e4_256_30 > ../Analysis/logs/train/step3/2h5h1_1e4_256_30.txt
#
echo "Evaluating model 2h5h1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h5h1 --model_name 2h5h1_1e4_256_30 > ../Analysis/logs/eval/step3/2h5h1_1e4_256_30.txt
#
#echo "Training model 2h1h6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_30 > ../Analysis/logs/train/step3/2h1h6k_1e4_256_30.txt
#
echo "Evaluating model 2h1h6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1h6k --model_name 2h1h6k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1h6k_1e4_256_30.txt
#
#
#echo "Training model 2h1h5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_30 > ../Analysis/logs/train/step3/2h1h5k_1e4_256_30.txt
#
echo "Evaluating model 2h1h5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1h5k --model_name 2h1h5k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1h5k_1e4_256_30.txt
#
#echo "Training model 2h1h4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_30 > ../Analysis/logs/train/step3/2h1h4k_1e4_256_30.txt
#
echo "Evaluating model 2h1h4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1h4k --model_name 2h1h4k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1h4k_1e4_256_30.txt
#
#echo "Training model 2h1h3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_30 > ../Analysis/logs/train/step3/2h1h3k_1e4_256_30.txt
#
echo "Evaluating model 2h1h3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1h3k --model_name 2h1h3k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1h3k_1e4_256_30.txt
#
#echo "Training model 2h1h2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_30 > ../Analysis/logs/train/step3/2h1h2k_1e4_256_30.txt
#
echo "Evaluating model 2h1h2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1h2k --model_name 2h1h2k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1h2k_1e4_256_30.txt
#
#echo "Training model 2h1h1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_30 > ../Analysis/logs/train/step3/2h1h1k_1e4_256_30.txt
#
echo "Evaluating model 2h1h1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1h1k --model_name 2h1h1k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1h1k_1e4_256_30.txt
#
#echo "Training model 2h1h5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1h5h --model_name 2h1h5h_1e4_256_30 > ../Analysis/logs/train/step3/2h1h5h_1e4_256_30.txt
#
echo "Evaluating model 2h1h5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1h5h --model_name 2h1h5h_1e4_256_30 > ../Analysis/logs/eval/step3/2h1h5h_1e4_256_30.txt
#
#echo "Training model 2h1h1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1h1h --model_name 2h1h1h_1e4_256_30 > ../Analysis/logs/train/step3/2h1h1h_1e4_256_30.txt
#
echo "Evaluating model 2h1h1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1h1h --model_name 2h1h1h_1e4_256_30 > ../Analysis/logs/eval/step3/2h1h1h_1e4_256_30.txt
#
#echo "Training model 2h1h10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1h10 --model_name 2h1h10_1e4_256_30 > ../Analysis/logs/train/step3/2h1h10_1e4_256_30.txt
#
echo "Evaluating model 2h1h10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1h10 --model_name 2h1h10_1e4_256_30 > ../Analysis/logs/eval/step3/2h1h10_1e4_256_30.txt
#
#echo "Training model 2h1h1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1h1 --model_name 2h1h1_1e4_256_30 > ../Analysis/logs/train/step3/2h1h1_1e4_256_30.txt
#
echo "Evaluating model 2h1h1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1h1 --model_name 2h1h1_1e4_256_30 > ../Analysis/logs/eval/step3/2h1h1_1e4_256_30.txt
#
#echo "Training model 2h10_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h10_6k --model_name 2h10_6k_1e4_256_30 > ../Analysis/logs/train/step3/2h10_6k_1e4_256_30.txt
#
echo "Evaluating model 2h10_6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_6k --model_name 2h10_6k_1e4_256_30 > ../Analysis/logs/eval/step3/2h10_6k_1e4_256_30.txt
#
#
#echo "Training model 2h10_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h10_5k --model_name 2h10_5k_1e4_256_30 > ../Analysis/logs/train/step3/2h10_5k_1e4_256_30.txt
#
echo "Evaluating model 2h10_5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_5k --model_name 2h10_5k_1e4_256_30 > ../Analysis/logs/eval/step3/2h10_5k_1e4_256_30.txt
#
#echo "Training model 2h10_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h10_4k --model_name 2h10_4k_1e4_256_30 > ../Analysis/logs/train/step3/2h10_4k_1e4_256_30.txt
#
echo "Evaluating model 2h10_4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_4k --model_name 2h10_4k_1e4_256_30 > ../Analysis/logs/eval/step3/2h10_4k_1e4_256_30.txt
#
#echo "Training model 2h10_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h10_3k --model_name 2h10_3k_1e4_256_30 > ../Analysis/logs/train/step3/2h10_3k_1e4_256_30.txt
#
echo "Evaluating model 2h10_3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_3k --model_name 2h10_3k_1e4_256_30 > ../Analysis/logs/eval/step3/2h10_3k_1e4_256_30.txt
#
#echo "Training model 2h10_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h10_2k --model_name 2h10_2k_1e4_256_30 > ../Analysis/logs/train/step3/2h10_2k_1e4_256_30.txt
#
echo "Evaluating model 2h10_2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_2k --model_name 2h10_2k_1e4_256_30 > ../Analysis/logs/eval/step3/2h10_2k_1e4_256_30.txt
#
#echo "Training model 2h10_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h10_1k --model_name 2h10_1k_1e4_256_30 > ../Analysis/logs/train/step3/2h10_1k_1e4_256_30.txt
#
echo "Evaluating model 2h10_1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_1k --model_name 2h10_1k_1e4_256_30 > ../Analysis/logs/eval/step3/2h10_1k_1e4_256_30.txt
#
#echo "Training model 2h10_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h10_5h --model_name 2h10_5h_1e4_256_30 > ../Analysis/logs/train/step3/2h10_5h_1e4_256_30.txt
#
echo "Evaluating model 2h10_5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_5h --model_name 2h10_5h_1e4_256_30 > ../Analysis/logs/eval/step3/2h10_5h_1e4_256_30.txt
#
#echo "Training model 2h10_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h10_1h --model_name 2h10_1h_1e4_256_30 > ../Analysis/logs/train/step3/2h10_1h_1e4_256_30.txt
#
echo "Evaluating model 2h10_1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_1h --model_name 2h10_1h_1e4_256_30 > ../Analysis/logs/eval/step3/2h10_1h_1e4_256_30.txt
#
#echo "Training model 2h10_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h10_10 --model_name 2h10_10_1e4_256_30 > ../Analysis/logs/train/step3/2h10_10_1e4_256_30.txt
#
echo "Evaluating model 2h10_10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_10 --model_name 2h10_10_1e4_256_30 > ../Analysis/logs/eval/step3/2h10_10_1e4_256_30.txt
#
#echo "Training model 2h10_1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h10_1 --model_name 2h10_1_1e4_256_30 > ../Analysis/logs/train/step3/2h10_1_1e4_256_30.txt
#
echo "Evaluating model 2h10_1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h10_1 --model_name 2h10_1_1e4_256_30 > ../Analysis/logs/eval/step3/2h10_1_1e4_256_30.txt
#
#echo "Training model 2h1_6k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1_6k --model_name 2h1_6k_1e4_256_30 > ../Analysis/logs/train/step3/2h1_6k_1e4_256_30.txt
#
echo "Evaluating model 2h1_6k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_6k --model_name 2h1_6k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1_6k_1e4_256_30.txt
#
#
#echo "Training model 2h1_5k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1_5k --model_name 2h1_5k_1e4_256_30 > ../Analysis/logs/train/step3/2h1_5k_1e4_256_30.txt
#
echo "Evaluating model 2h1_5k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_5k --model_name 2h1_5k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1_5k_1e4_256_30.txt
#
#echo "Training model 2h1_4k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1_4k --model_name 2h1_4k_1e4_256_30 > ../Analysis/logs/train/step3/2h1_4k_1e4_256_30.txt
#
echo "Evaluating model 2h1_4k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_4k --model_name 2h1_4k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1_4k_1e4_256_30.txt
#
#echo "Training model 2h1_3k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1_3k --model_name 2h1_3k_1e4_256_30 > ../Analysis/logs/train/step3/2h1_3k_1e4_256_30.txt
#
echo "Evaluating model 2h1_3k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_3k --model_name 2h1_3k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1_3k_1e4_256_30.txt
#
#echo "Training model 2h1_2k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1_2k --model_name 2h1_2k_1e4_256_30 > ../Analysis/logs/train/step3/2h1_2k_1e4_256_30.txt
#
echo "Evaluating model 2h1_2k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_2k --model_name 2h1_2k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1_2k_1e4_256_30.txt
#
#echo "Training model 2h1_1k, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1_1k --model_name 2h1_1k_1e4_256_30 > ../Analysis/logs/train/step3/2h1_1k_1e4_256_30.txt
#
echo "Evaluating model 2h1_1k_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_1k --model_name 2h1_1k_1e4_256_30 > ../Analysis/logs/eval/step3/2h1_1k_1e4_256_30.txt
#
#echo "Training model 2h1_5h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1_5h --model_name 2h1_5h_1e4_256_30 > ../Analysis/logs/train/step3/2h1_5h_1e4_256_30.txt
#
echo "Evaluating model 2h1_5h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_5h --model_name 2h1_5h_1e4_256_30 > ../Analysis/logs/eval/step3/2h1_5h_1e4_256_30.txt
#
#echo "Training model 2h1_1h, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1_1h --model_name 2h1_1h_1e4_256_30 > ../Analysis/logs/train/step3/2h1_1h_1e4_256_30.txt
#
echo "Evaluating model 2h1_1h_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_1h --model_name 2h1_1h_1e4_256_30 > ../Analysis/logs/eval/step3/2h1_1h_1e4_256_30.txt
#
#echo "Training model 2h1_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1_10 --model_name 2h1_10_1e4_256_30 > ../Analysis/logs/train/step3/2h1_10_1e4_256_30.txt
#
echo "Evaluating model 2h1_10_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_10 --model_name 2h1_10_1e4_256_30 > ../Analysis/logs/eval/step3/2h1_10_1e4_256_30.txt
#
#echo "Training model 2h1_1, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --patience 30 --model_folder step3 \
#              --train_path $trn --val_path $val      \
#              --n_epochs 5 --lr 1e-4 --batch_size 256 \
#              --classifier 2h1_1 --model_name 2h1_1_1e4_256_30 > ../Analysis/logs/train/step3/2h1_1_1e4_256_30.txt
#
echo "Evaluating model 2h1_1_1e4_256"
python ../eval.py --train_path $trn --test_path $tst \
              --model_folder step3 \
              --classifier 2h1_1 --model_name 2h1_1_1e4_256_30 > ../Analysis/logs/eval/step3/2h1_1_1e4_256_30.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../traineval2excel.py --xls_name 'ANN_step3' --archives_folder 'step3'

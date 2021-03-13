#!/bin/bash

mkdir -p ../Results/Training/logs/separated
mkdir -p ../Results/Testing/logs/separated
mkdir -p ../models/separated

trn="Train_data_v3.hdf5"
val="Validation_data_v3.hdf5"

stead_seis_tst="Test_stead_seis.hdf5"
stead_nseis_tst="Test_stead_noise.hdf5"
geo_test="Test_geo.hdf5"

#echo "Training model Cnn1_3k_10, lr = 1e-4, epochs = 5, batch_size = 256"
#python ../train_validation.py \
#              --lr 1e-4  \
#              --epochs 5 \
#              --patience 40 \
#              --batch_size 256 \
#              --model_folder separated \
#              --train_path $trn \
#              --val_path $val \
#              --classifier Cnn1_3k_10 \
#              --model_name Cnn1_3k_10_1e4_256_40 > ../Results/Training/logs/separated/Cnn1_3k_10_1e4_256_40.txt

echo "Evaluating model Cnn1_3k_10_1e4_256"
python ../eval_separated.py \
              --train_path $trn \
              --stead_seis_test_path $stead_seis_tst \
              --stead_nseis_test_path $stead_nseis_tst \
              --geo_test_path $geo_test \
              --model_folder separated \
              --classifier Cnn1_3k_10 \
              --model_name Cnn1_3k_10_1e4_256_40 > ../Results/Testing/logs/separated/Cnn1_3k_10_1e4_256_40.txt

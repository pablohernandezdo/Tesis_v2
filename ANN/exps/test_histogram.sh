#!/bin/bash

mkdir -p ../Analysis/logs/tst_hist
mkdir -p ../Analysis/logs/tst_hist

tst="Test_data.hdf5"
trn="Train_data.hdf5"
val="Validation_data.hdf5"

echo "Evaluating model 2h5h5k_1e3_256"
python ../eval_curves.py --test_path $tst \
              --model_folder step4 \
              --classifier 2h5h5k --model_name 2h5h5k_1e3_256_20 > ../Analysis/logs/tst_hist/2h5h5k_1e3_256_20.txt

#!/bin/bash

cd ..
mkdir -p logs/CBN_v2/train
mkdir -p models

trn="Train_data.hdf5"
val="Validation_data.hdf5"

# Train CBN_v2 with evaluation for 1 epoch
echo "Training CBN model on $trn and validating on $val"
python train_validation.py --train_path $trn --val_path $val \
              --classifier CBN_v2 --model_name CBN_v2_1epch_train_validation \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > logs/CBN_v2/train/CBN_v2_1epch_validation.txt
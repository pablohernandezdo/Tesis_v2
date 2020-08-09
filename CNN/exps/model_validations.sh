#!/bin/bash

mkdir -p ../logs/CBN/train
mkdir -p ../models

trn="Train_data.hdf5"
val="Validation_data.hdf5"

# Train cbn1epch model with evaluation split for 1 epoch
echo "Training CBN model on $trn and validating on $val"
python ../train_validation.py --train_path $trn --val_path $val  \
              --classifier CBN --model_name CBN_1epch_validation \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/CBN/train/CBN_1epch_validation.txt

# Train cbn10epch model with evaluation split for 10 epochs
echo "Training CBN model on $trn and validating on $val"
python ../train_validation.py --train_path $trn --val_path $val \
            --classifier CBN --model_name CBN_10epch_validation \
            --n_epochs 10 --batch_size 32 --lr 1e-6  > ../logs/CBN/train/CBN_10epch_validation.txt

# Train cbn1epch_v2 model with evaluation split for 1 epoch
echo "Training CBN model on $trn and validating on $val"
python ../train_validation.py --train_path $trn --val_path $val \
              --classifier CBN_v2 --model_name CBN_v2_1epch_train_validation \
              --n_epochs 1 --batch_size 32 --lr 1e-6  > ../logs/CBN_v2/train/CBN_v2_1epch_validation.txt

# Train cbn10epch_v2 model with evaluation split for 10 epochs
echo "Training CBN model on $trn and validating on $val"
python ../train_validation.py --train_path $trn --val_path $val \
              --classifier CBN_v2 --model_name CBN_v2_10epch_train_validation \
              --n_epochs 10 --batch_size 32 --lr 1e-6  > ../logs/CBN_v2/train/CBN_v2_10epch_validation.txt
#!/bin/bash

mkdir -p ../Analysis/logs/

mf="step4"

# Test dataset inference on Cnn1_2k_1h_5e5_256_40 model
echo "Running DAS test dataset inference on Cnn1_2k_1h_5e5_256_40 model"
python ../inf_california_fotos.py --model_folder $mf --model_name Cnn1_2k_1h_5e5_256_40 --classifier Cnn1_2k_1h


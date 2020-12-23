#!/bin/bash

fld="step4"
mdl="Cnn1_1k_2k_1e4_256_20"

echo Saving histogram model
python ../output2histogram.py --model_folder $fld --model_name $mdl --n_bins 100
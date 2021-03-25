#!/bin/bash

mkdir -p ../Results/Testing/Logs/DAS_sep

das_seis="DAS_seismic.hdf5"
das_nseis="DAS_non_seismic.hdf5"
das_noise="DAS_noise.hdf5"

echo "Evaluating model Cnn1_3k_10_1e4_256"
python ../eval_das_sep_extremos.py \
              --das_seis_path $das_seis \
              --das_nseis_path $das_nseis \
              --das_noise_path $das_noise \
              --classifier Cnn1_3k_10 \
              --model_folder separated \
              --model_name Cnn1_3k_10_1e4_256_40 > ../Results/Testing/Logs/DAS_sep/Cnn1_3k_10_1e4_256_40.txt

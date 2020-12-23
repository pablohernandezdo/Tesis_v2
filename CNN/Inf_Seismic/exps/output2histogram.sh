#!/bin/bash

fld="step4"

python ../output2histogram.py --model_folder $fld --model_name "Cnn1_1k_2k_1e4_256_20" --n_bins 30

python ../output2histogram.py --model_folder $fld --model_name "Cnn1_2k_1h_5e4_256_40" --n_bins 30

python ../output2histogram.py --model_folder $fld --model_name "Cnn1_2k_1h_1e4_256_30" --n_bins 30

python ../output2histogram.py --model_folder $fld --model_name "Cnn1_2k_1h_5e5_256_40" --n_bins 30

python ../output2histogram.py --model_folder $fld --model_name "Cnn1_3k_10_1e4_256_25" --n_bins 30

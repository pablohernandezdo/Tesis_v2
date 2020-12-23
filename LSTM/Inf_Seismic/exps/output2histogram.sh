#!/bin/bash

fld="step5"

python ../output2histogram.py --model_folder $fld --model_name "Lstm_128_32_1_1_5e4_256_20" --n_bins 30

python ../output2histogram.py --model_folder $fld --model_name "Lstm_128_32_2_1_5e4_256_35" --n_bins 30

python ../output2histogram.py --model_folder $fld --model_name "Lstm_128_32_2_1_5e4_256_25" --n_bins 30

python ../output2histogram.py --model_folder $fld --model_name "Lstm_128_32_2_1_5e4_256_30" --n_bins 30

# python ../output2histogram.py --model_folder $fld --model_name "Lstm_128_32_1_1_5e4_256_35" --n_bins 30

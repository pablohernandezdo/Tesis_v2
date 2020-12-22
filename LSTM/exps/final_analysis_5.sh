#!/bin/bash

# Get best models PR, ROC and Fscore curves
python ../final_analysis.py --step4_folder 'step4' \
                            --step5_folder 'step5' \
                            --best_models "Lstm_128_32_1_1_5e4_256_20.txt Lstm_128_32_1_1_5e4_256_35.txt Lstm_128_32_2_1_5e4_256_25.txt Lstm_128_32_2_1_5e4_256_30.txt Lstm_128_32_2_1_5e4_256_35.txt"

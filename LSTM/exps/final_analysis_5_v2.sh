#!/bin/bash

# Get best models PR, ROC and Fscore curves
python ../final_analysis.py --step4_folder 'step4' \
                            --step5_folder 'step5_v2' \
                            --best_models "Lstm_16_16_1_1_5e3_256_35.txt Lstm_64_64_5_1_1e2_256_25.txt Lstm_32_64_1_1_5e4_256_35.txt Lstm_128_32_2_1_5e3_256_25.txt Lstm_32_64_1_1_1e2_256_20.txt"

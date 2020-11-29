#!/bin/bash

# Get best models PR, ROC and Fscore curves
python ../final_analysis.py --step4_folder 'step4' \
                            --step5_folder 'step5' \
                            --best_models "Cnn1_2k_1h_5e5_256_40.txt Cnn1_3k_10_1e4_256_40.txt Cnn1_3k_10_1e4_256_25.txt Cnn1_3k_10_1e4_256_40.txt Cnn1_2k_1h_5e5_256_40.txt Cnn1_6k_2h_1e4_256_30.txt Cnn1_3k_10_1e4_256_25.txt Cnn1_3k_10_1e4_256_40.txt Cnn1_2k_1h_5e5_256_40.txt Cnn1_6k_2h_1e4_256_30.txt" \

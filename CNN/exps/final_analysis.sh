#!/bin/bash

# Get best models PR, ROC and Fscore curves
python ../final_analysis.py --step4_folder 'step4' \
                            --step5_folder 'step5' \
                            --avg_folder 'avgmodels' \
                            --best_models "Cnn1_2k_1h_5e5_256_40.txt Cnn1_3k_10_1e4_256_40.txt Cnn1_3k_10_1e4_256_25.txt Cnn1_3k_10_1e4_256_40.txt Cnn1_2k_1h_5e5_256_40.txt Cnn1_6k_2h_1e4_256_30.txt Cnn1_3k_10_1e4_256_25.txt Cnn1_3k_10_1e4_256_40.txt Cnn1_2k_1h_5e5_256_40.txt Cnn1_6k_2h_1e4_256_30.txt" \
                            # --avg_models "2h1h10_1e4_256_30.txt 1h5h_1e4_256_30.txt 2h10_2k_1e4_256_30.txt 2h1_2k_1e4_256_30.txt 2h5h10_1e4_256_30.txt 2h1h1h_1e4_256_30.txt 2h10_1k_1e4_256_30.txt 1h1k_1e4_256_30.txt 2h5h1h_1e4_256_30.txt 2h1h1k_1e4_256_30.txt"
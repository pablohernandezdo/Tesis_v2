#!/bin/bash

# Get best models PR, ROC and Fscore curves
python ../final_analysis.py --step4_folder 'step4' \
                            --step5_folder 'step5' \
                            --avg_folder 'avgmodels' \
                            --best_models "2h2k5k_5e5_256_25.txt 2h1k5k_5e5_256_20.txt 2h1k6k_5e5_256_20.txt 2h1k4k_5e5_256_25.txt 2h1k6k_5e5_256_30.txt 2h2k5k_5e5_256_20.txt 2h5h4k_5e5_256_35.txt 2h1k6k_1e4_256_20.txt 2h5h6k_5e5_256_25.txt 2h1k4k_1e4_256_20.txt" \
                            --avg_models "2h1h10_1e4_256_30.txt 1h5h_1e4_256_30.txt 2h10_2k_1e4_256_30.txt 2h1_2k_1e4_256_30.txt 2h5h10_1e4_256_30.txt 2h1h1h_1e4_256_30.txt 2h10_1k_1e4_256_30.txt 1h1k_1e4_256_30.txt 2h5h1h_1e4_256_30.txt 2h1h1k_1e4_256_30.txt"
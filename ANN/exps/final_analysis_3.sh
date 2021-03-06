#!/bin/bash

# Get best models PR, ROC and Fscore curves
python ../final_analysis.py --step4_folder 'step4' \
                            --step5_folder 'step5' \
                            --avg_folder 'avgmodels' \
                            --avg_das_folder 'avgmodels_das' \
                            --best_models "2h2k5k_5e5_256_25.txt 2h1k5k_5e5_256_20.txt 2h1k6k_5e5_256_20.txt" \
                            --avg_models "2h1h10_1e4_256_30.txt 1h5h_1e4_256_30.txt 2h10_2k_1e4_256_30.txt"
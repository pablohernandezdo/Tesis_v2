#!/bin/bash

# Get best models PR, ROC and Fscore curves
python ../bestmodel_curves.py --archives_folder 'step4_curves' \
                              --best_models "2h2k5k_5e5_256_25.txt 2h1k5k_5e5_256_20.txt 2h1k6k_5e5_256_20.txt 2h1k4k_5e5_256_25.txt 2h1k6k_5e5_256_30.txt 2h2k5k_5e5_256_20.txt 2h5h4k_5e5_256_35.txt 2h1k6k_1e4_256_20.txt 2h5h6k_5e5_256_25.txt 2h1k4k_1e4_256_20.txt"
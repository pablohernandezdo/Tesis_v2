#!/bin/bash

mdl="2h2k5k_5e5_256_25.txt"
mdl="mdl 2h1k5k_5e5_256_20.txt"
mdl="mdl 2h1k6k_5e5_256_20.txt"
mdl="mdl 2h1k4k_5e5_256_25.txt"
mdl="mdl 2h1k6k_5e5_256_30.txt"
mdl="mdl 2h2k5k_5e5_256_20.txt"
mdl="mdl 2h5h4k_5e5_256_35.txt"
mdl="mdl 2h1k6k_1e4_256_20.txt"
mdl="mdl 2h5h6k_5e5_256_25.txt"
mdl="mdl 22h1k4k_1e4_256_20.txt"

# Get best models PR, ROC and Fscore curves
python ../bestmodel_curves.py --archives_folder 'step4_curves' \
                              --best_models $mdl
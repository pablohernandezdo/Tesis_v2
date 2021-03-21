#!/bin/bash

echo "Building train, test and validation datasets"
#python new_createh5.py --train_traces 102400 --train_noise 51200 \
#                       --test_traces 12800 --test_noise 6400   \
#                       --val_traces 12800 --val_noise 6400

python new_createh5.py --train_traces 5120 --train_noise 2560 \
                       --test_traces 2048 --test_noise 1024   \
                       --val_traces 2048 --val_noise 1024

#!/bin/bash

echo "Building train, test and validation datasets"
python createh5_v2.py --train_traces 102400 --train_noise 51200 \
                      --test_traces 12800 --test_noise 6400   \
                      --val_traces 12800 --val_noise 6400

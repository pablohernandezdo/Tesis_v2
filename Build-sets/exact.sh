#!/bin/bash

echo "Building train, test and validation datasets"
python createh5_v2.py --train_traces 38400 --train_noise 38400 \
                      --test_traces 7680 --test_noise 7680   \
                      --val_traces 7680 --val_noise 7680

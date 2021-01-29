#!/bin/bash

echo "Building train, test and validation datasets"
python createh5_v2.py --train_traces 40960 --train_noise 40960 \
                      --test_traces 5120 --test_noise 5120   \
                      --val_traces 5120 --val_noise 5120

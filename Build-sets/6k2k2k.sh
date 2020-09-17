#!/bin/bash

echo "Building train, test and validation datasets"
python createh5_v2.py --train_traces 3000 --train_noise 3000 \
                      --test_traces 1000 --test_noise 1000   \
                      --val_traces 1000 --val_noise 1000

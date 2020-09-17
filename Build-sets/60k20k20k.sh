#!/bin/bash

echo "Building train, test and validation datasets"
python createh5_v2.py --train_traces 30000 --train_noise 30000 \
                      --test_traces 10000 --test_noise 10000   \
                      --val_traces 10000 --val_noise 10000

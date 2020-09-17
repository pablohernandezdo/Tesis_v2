#!/bin/bash

echo "Building train, test and validation datasets"
python createh5_v2.py --train_traces 5120 --train_noise 5120 \
                      --test_traces 2048 --test_noise 2048   \
                      --val_traces 2048 --val_noise 2048

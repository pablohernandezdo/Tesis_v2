#!/bin/bash

echo "Building train, test and validation datasets"
python createh5_v2.py --train_traces 10240 --train_noise 10240 \
                      --test_traces 2048 --test_noise 2048   \
                      --val_traces 1024 --val_noise 1024

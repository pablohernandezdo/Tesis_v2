#!/bin/bash

echo "Building train, test and validation datasets"
python createh5_v2.py --train_traces 71680 --train_noise 71680 \
                      --test_traces 15360 --test_noise 15360   \
                      --val_traces 15360 --val_noise 15360

#!/bin/bash

echo "Building train, test and validation datasets"
python createh5_sep.py --train_traces 52500 --train_noise 26250 \
                       --test_traces 10000 --test_noise 10000   \
                       --val_traces 7500 --val_noise 3750


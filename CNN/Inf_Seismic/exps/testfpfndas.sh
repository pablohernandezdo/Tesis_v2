#!/bin/bash

mkdir -p ../logs/

# Test dataset inference on 2c3k3k_1e3_256 model
echo "Running DAS test dataset inference on 2c3k3k_1e3_256 model"
python ../inf_test_dataset.py --model_name 2c3k3k_1e3_256 --classifier 2c3k3k  > ../logs/2c3k3k_1e3_256.txt
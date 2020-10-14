#!/bin/bash

mkdir -p ../logs/

# Test dataset inference on 2h5h5k_1e4_256_30 model
echo "Running DAS test dataset inference on 1c1h_1e6_256_30 model"
python ../inf_test_dataset.py --model_name 1c1h_1e6_256_30 --classifier 1c1h  > ../logs/1c1h_1e6_256_30.txt



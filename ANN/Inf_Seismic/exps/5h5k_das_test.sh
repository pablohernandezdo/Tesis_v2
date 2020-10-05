#!/bin/bash

mkdir -p ../logs/inf

# Test dataset inference on 2h5h5k_1e4_256model
echo "Running DAS test dataset inference on 2h5h5k_1e4_256 model"
python ../inf_test_dataset.py --model_name 2h5h5k_1e4_256 --classifier 2h5h5k  > ../logs/inf/2h5h5k_1e4_256.txt
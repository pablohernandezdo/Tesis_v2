#!/bin/bash

mkdir -p ../logs/CBN/inf

# Test dataset inference on cbn1epch model
echo "Running DAS test dataset inference on cbn1epch model"
python ../inf_test_dataset.py --model_name CBN_1epch --classifier CBN  > ../logs/CBN/inf/cbn1epch.txt
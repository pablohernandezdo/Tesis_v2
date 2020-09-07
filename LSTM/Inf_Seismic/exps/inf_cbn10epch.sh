#!/bin/bash

mkdir -p ../logs/CBN/inf

# Test dataset inference on cbn10epch model
echo "Running DAS test dataset inference on cbn10epch model"
python ../inf_test_dataset.py --model_name CBN_10epch --classifier CBN  > ../logs/CBN/inf/cbn10epch.txt

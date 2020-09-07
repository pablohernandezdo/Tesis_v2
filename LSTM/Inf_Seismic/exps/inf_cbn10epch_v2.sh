#!/bin/bash

mkdir -p ../logs/CBN_v2/inf

# Test dataset inference on cbn10epch_v2 model
echo "Running DAS test dataset inference on cbn10epch_v2 model"
python ../inf_test_dataset.py --model_name CBN_v2_10epch --classifier CBN_v2  > ../logs/CBN_v2/inf/cbn10epch_v2.txt

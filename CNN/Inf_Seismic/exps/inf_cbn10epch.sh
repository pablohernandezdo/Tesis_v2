#!/bin/bash

mkdir -p ../logs/CBN/inf

# Full inference on cbn1epch model
echo "Running DAS test dataset inference on cbn10epch model"
python ../inf_full.py --model_name CBN_10epch --classifier CBN  > ../logs/CBN/inf/cbn10epch.txt
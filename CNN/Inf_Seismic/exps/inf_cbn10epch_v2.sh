#!/bin/bash

mkdir -p ../logs/CBN_v2/inf

# Full inference on cbn1epch model
echo "Running DAS test dataset inference on cbn10epch_v2 model"
python ../inf_full.py --model_name CBN_v2_10epch --classifier CBN_v2  > ../logs/CBN/inf/cbn10epch_v2.txt

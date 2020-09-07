#!/bin/bash

mkdir -p ../logs/LSTM/inf

# Test dataset inference on cbn1epch model
echo "Running DAS test dataset inference on lstm_1e4_64 model"
python ../inf_test_dataset.py --model_name C --classifier lstm_1e4_64_validation  > ../logs/LSTM/inf/lstm_1e4_64_validation.txt

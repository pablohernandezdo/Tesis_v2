#!/bin/bash

mkdir -p ../logs/LSTM/inf

# Test dataset inference on cbn1epch model
echo "Running DAS test dataset inference on lstm_1e4_64 model"
python ../inf_test_dataset.py --model_name LSTM_5epch_1e4_32 --classifier LSTM  > ../logs/LSTM/inf/LSTM_5epch_1e4_32.txt

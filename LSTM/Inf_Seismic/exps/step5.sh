#!/bin/bash

mkdir -p ../Analysis/logs/itd

mf="step5"

# Test dataset inference on Lstm_32_64_1_1_1e3_256_40 model
echo "Running DAS test dataset inference on Lstm_32_64_1_1_1e3_256_40 model"
python ../inf_test_dataset.py --model_folder $mf --model_name Lstm_32_64_1_1_1e3_256_40 --classifier Lstm_32_64_1_1  > ../Analysis/logs/itd/Lstm_32_64_1_1_1e3_256_40.txt

# Test dataset inference on Lstm_16_16_1_1_1e3_256_40 model
echo "Running DAS test dataset inference on Lstm_16_16_1_1_1e3_256_40 model"
python ../inf_test_dataset.py --model_folder $mf --model_name Lstm_16_16_1_1_1e3_256_40 --classifier Lstm_16_16_1_1  > ../Analysis/logs/itd/Lstm_16_16_1_1_1e3_256_40.txt

## Test dataset inference on Lstm_16_16_1_1_1e3_256_40 model
#echo "Running DAS test dataset inference on Lstm_16_16_1_1_1e3_256_40 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name Lstm_16_16_1_1_1e3_256_40 --classifier Lstm_16_16_1_1  > ../Analysis/logs/itd/Lstm_16_16_1_1_1e3_256_40.txt

# Test dataset inference on Lstm_128_32_1_1_5e4_256_20 model
echo "Running DAS test dataset inference on Lstm_128_32_1_1_5e4_256_20 model"
python ../inf_test_dataset.py --model_folder $mf --model_name Lstm_128_32_1_1_5e4_256_20 --classifier Lstm_128_32_1_1  > ../Analysis/logs/itd/Lstm_128_32_1_1_5e4_256_20.txt

## Test dataset inference on Lstm_128_32_1_1_5e4_256_20 model
#echo "Running DAS test dataset inference on Lstm_128_32_1_1_5e4_256_20 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name Lstm_128_32_1_1_5e4_256_20 --classifier Lstm_128_32_1_1  > ../Analysis/logs/itd/Lstm_128_32_1_1_5e4_256_20.txt

## Test dataset inference on Lstm_16_16_1_1_1e3_256_40 model
#echo "Running DAS test dataset inference on Lstm_16_16_1_1_1e3_256_40 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name Lstm_16_16_1_1_1e3_256_40 --classifier Lstm_16_16_1_1  > ../Analysis/logs/itd/Lstm_16_16_1_1_1e3_256_40.txt

# Test dataset inference on Lstm_128_32_2_1_5e4_256_35 model
echo "Running DAS test dataset inference on Lstm_128_32_2_1_5e4_256_35 model"
python ../inf_test_dataset.py --model_folder $mf --model_name Lstm_128_32_2_1_5e4_256_35 --classifier Lstm_128_32_2_1  > ../Analysis/logs/itd/Lstm_128_32_2_1_5e4_256_35.txt

## Test dataset inference on Lstm_16_16_1_1_1e3_256_40 model
#echo "Running DAS test dataset inference on Lstm_16_16_1_1_1e3_256_40 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name Lstm_16_16_1_1_1e3_256_40 --classifier Lstm_16_16_1_1  > ../Analysis/logs/itd/Lstm_16_16_1_1_1e3_256_40.txt

# Test dataset inference on Lstm_32_32_2_1_5e4_256_20 model
echo "Running DAS test dataset inference on Lstm_32_32_2_1_5e4_256_20 model"
python ../inf_test_dataset.py --model_folder $mf --model_name Lstm_32_32_2_1_5e4_256_20 --classifier Lstm_32_32_2_1  > ../Analysis/logs/itd/Lstm_32_32_2_1_5e4_256_20.txt

## Test dataset inference on Lstm_128_32_1_1_5e4_256_20 model
#echo "Running DAS test dataset inference on Lstm_128_32_1_1_5e4_256_20 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name Lstm_128_32_1_1_5e4_256_20 --classifier Lstm_128_32_1_1  > ../Analysis/logs/itd/Lstm_128_32_1_1_5e4_256_20.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../dastest2excel.py --xls_name 'LSTM_step5' --archives_folder 'itd'
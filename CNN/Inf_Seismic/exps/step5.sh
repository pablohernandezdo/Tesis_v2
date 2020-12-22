#!/bin/bash

mkdir -p ../Analysis/logs/

mf="step4"

## Test dataset inference on Cnn1_3k_10_1e4_256_40 model
#echo "Running DAS test dataset inference on Cnn1_3k_10_1e4_256_40 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name Cnn1_3k_10_1e4_256_40 --classifier Cnn1_3k_10  > ../Analysis/logs/Cnn1_3k_10_1e4_256_40.txt
#
## Test dataset inference on Cnn1_3k_1e3_256_30 model
#echo "Running DAS test dataset inference on Cnn1_3k_1e3_256_30 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name Cnn1_3k_1e3_256_30 --classifier Cnn1_3k  > ../Analysis/logs/Cnn1_3k_1e3_256_30.txt
#
# Test dataset inference on Cnn1_2k_1h_5e4_256_40 model
echo "Running DAS test dataset inference on Cnn1_2k_1h_5e4_256_40 model"
python ../inf_test_dataset.py --model_folder $mf --model_name Cnn1_2k_1h_5e4_256_40 --classifier Cnn1_2k_1h  > ../Analysis/logs/Cnn1_2k_1h_5e4_256_40.txt
#
## Test dataset inference on Cnn1_6k_5k_1e4_256_30 model
#echo "Running DAS test dataset inference on Cnn1_6k_5k_1e4_256_30 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name Cnn1_6k_5k_1e4_256_30 --classifier Cnn1_6k_5k  > ../Analysis/logs/Cnn1_6k_5k_1e4_256_30.txt
#
## Test dataset inference on Cnn1_6k_5k_1e4_256_25 model
#echo "Running DAS test dataset inference on Cnn1_6k_5k_1e4_256_25 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name Cnn1_6k_5k_1e4_256_25 --classifier Cnn1_6k_5k  > ../Analysis/logs/Cnn1_6k_5k_1e4_256_25.txt
#
# Test dataset inference on Cnn1_2k_1h_1e4_256_30 model
echo "Running DAS test dataset inference on Cnn1_2k_1h_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name Cnn1_2k_1h_1e4_256_30 --classifier Cnn1_2k_1h  > ../Analysis/logs/Cnn1_2k_1h_1e4_256_30.txt
#
# Test dataset inference on Cnn1_1k_2k_1e4_256_20 model
echo "Running DAS test dataset inference on Cnn1_1k_2k_1e4_256_20 model"
python ../inf_test_dataset.py --model_folder $mf --model_name Cnn1_1k_2k_1e4_256_20 --classifier Cnn1_1k_2k  > ../Analysis/logs/Cnn1_1k_2k_1e4_256_20.txt
#
# Test dataset inference on Cnn1_2k_1h_5e5_256_40 model
echo "Running DAS test dataset inference on Cnn1_2k_1h_5e5_256_40 model"
python ../inf_test_dataset.py --model_folder $mf --model_name Cnn1_2k_1h_5e5_256_40 --classifier Cnn1_2k_1h  > ../Analysis/logs/Cnn1_2k_1h_5e5_256_40.txt
#
# Test dataset inference on Cnn1_3k_10_1e4_256_25 model
echo "Running DAS test dataset inference on Cnn1_3k_10_1e4_256_25 model"
python ../inf_test_dataset.py --model_folder $mf --model_name Cnn1_3k_10_1e4_256_25 --classifier Cnn1_3k_10  > ../Analysis/logs/Cnn1_3k_10_1e4_256_25.txt
#
## Test dataset inference on Cnn1_6k_2h_1e4_256_30 model
#echo "Running DAS test dataset inference on Cnn1_6k_2h_1e4_256_30 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name Cnn1_6k_2h_1e4_256_30 --classifier Cnn1_6k_2h  > ../Analysis/logs/Cnn1_6k_2h_1e4_256_30.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../dastest2excel.py --xls_name 'CNN_step5' --archives_folder ''
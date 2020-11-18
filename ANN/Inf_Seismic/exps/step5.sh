#!/bin/bash

mkdir -p ../Analysis/logs/itd

mf="step4"

## Test dataset inference on 2h2k5k_5e5_256_25 model
#echo "Running DAS test dataset inference on 2h2k5k_5e5_256_25 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name 2h2k5k_5e5_256_25 --classifier 2h2k5k  > ../Analysis/logs/itd/2h2k5k_5e5_256_25.txt
#
## Test dataset inference on 2h2k5k_5e5_256_25 model
#echo "Running DAS test dataset inference on 2h2k5k_5e5_256_25 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name 2h2k5k_5e5_256_25 --classifier 2h2k5k  > ../Analysis/logs/itd/2h2k5k_5e5_256_25.txt
#
## Test dataset inference on 2h1k6k_5e5_256_20 model
#echo "Running DAS test dataset inference on 2h1k6k_5e5_256_20 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name 2h1k6k_5e5_256_20 --classifier 2h1k6k  > ../Analysis/logs/itd/2h1k6k_5e5_256_20.txt
#
## Test dataset inference on 2h1k4k_5e5_256_25 model
#echo "Running DAS test dataset inference on 2h1k4k_5e5_256_25 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name 2h1k4k_5e5_256_25 --classifier 2h1k4k  > ../Analysis/logs/itd/2h1k4k_5e5_256_25.txt
#
## Test dataset inference on 2h1k6k_5e5_256_30 model
#echo "Running DAS test dataset inference on 2h1k6k_5e5_256_30 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name 2h1k6k_5e5_256_30 --classifier 2h1k6k  > ../Analysis/logs/itd/2h1k6k_5e5_256_30.txt
#
## Test dataset inference on 2h2k5k_5e5_256_20 model
#echo "Running DAS test dataset inference on 2h2k5k_5e5_256_20 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name 2h2k5k_5e5_256_20 --classifier 2h2k5k  > ../Analysis/logs/itd/2h2k5k_5e5_256_20.txt
#
## Test dataset inference on 2h5h4k_5e5_256_35 model
#echo "Running DAS test dataset inference on 2h5h4k_5e5_256_35 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name 2h5h4k_5e5_256_35 --classifier 2h5h4k  > ../Analysis/logs/itd/2h5h4k_5e5_256_35.txt
#
## Test dataset inference on 2h1k6k_1e4_256_20 model
#echo "Running DAS test dataset inference on 2h1k6k_1e4_256_20 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name 2h1k6k_1e4_256_20 --classifier 2h1k6k  > ../Analysis/logs/itd/2h1k6k_1e4_256_20.txt
#
## Test dataset inference on 2h5h6k_5e5_256_25 model
#echo "Running DAS test dataset inference on 2h5h6k_5e5_256_25 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name 2h5h6k_5e5_256_25 --classifier 2h5h6k  > ../Analysis/logs/itd/2h5h6k_5e5_256_25.txt
#
## Test dataset inference on 2h1k4k_1e4_256_20 model
#echo "Running DAS test dataset inference on 2h1k4k_1e4_256_20 model"
#python ../inf_test_dataset.py --model_folder $mf --model_name 2h1k4k_1e4_256_20 --classifier 2h1k4k  > ../Analysis/logs/itd/2h1k4k_1e4_256_20.txt

# reports 2 excel

echo "Creating summary of reports excel file"
python ../dastest2excel.py --xls_name 'ANN_step5' --archives_folder 'itd'
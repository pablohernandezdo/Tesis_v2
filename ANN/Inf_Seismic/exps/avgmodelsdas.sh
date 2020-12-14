#!/bin/bash

mkdir -p ../Analysis/logs/avg

mf="avg"

# Test dataset inference on 2h1h10_1e4_256_30 model
echo "Running DAS test dataset inference on 2h1h10_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 2h1h10_1e4_256_30 --classifier 2h1h10  > ../Analysis/logs/avg/2h1h10_1e4_256_30.txt

# Test dataset inference on 1h5h_1e4_256_30 model
echo "Running DAS test dataset inference on 1h5h_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 1h5h_1e4_256_30 --classifier 1h5h  > ../Analysis/logs/avg/1h5h_1e4_256_30.txt

# Test dataset inference on 2h10_2k_1e4_256_30 model
echo "Running DAS test dataset inference on 2h10_2k_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 2h10_2k_1e4_256_30 --classifier 2h10_2k  > ../Analysis/logs/avg/2h10_2k_1e4_256_30.txt

# Test dataset inference on 2h1_2k_1e4_256_30 model
echo "Running DAS test dataset inference on 2h1_2k_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 2h1_2k_1e4_256_30 --classifier 2h1_2k  > ../Analysis/logs/avg/2h1_2k_1e4_256_30.txt

# Test dataset inference on 2h5h10_1e4_256_30 model
echo "Running DAS test dataset inference on 2h5h10_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 2h5h10_1e4_256_30 --classifier 2h5h10  > ../Analysis/logs/avg/2h5h10_1e4_256_30.txt

# Test dataset inference on 2h1h1h_1e4_256_30 model
echo "Running DAS test dataset inference on 2h1h1h_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 2h1h1h_1e4_256_30 --classifier 2h1h1h  > ../Analysis/logs/avg/2h1h1h_1e4_256_30.txt

# Test dataset inference on 2h10_1k_1e4_256_30 model
echo "Running DAS test dataset inference on 2h10_1k_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 2h10_1k_1e4_256_30 --classifier 2h10_1k  > ../Analysis/logs/avg/2h10_1k_1e4_256_30.txt

# Test dataset inference on 1h1k_1e4_256_30 model
echo "Running DAS test dataset inference on 1h1k_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 1h1k_1e4_256_30 --classifier 1h1k  > ../Analysis/logs/avg/1h1k_1e4_256_30.txt

# Test dataset inference on 2h5h1h_1e4_256_30 model
echo "Running DAS test dataset inference on 2h5h1h_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 2h5h1h_1e4_256_30 --classifier 2h5h1h  > ../Analysis/logs/avg/2h5h1h_1e4_256_30.txt

# Test dataset inference on 2h1h1k_1e4_256_30 model
echo "Running DAS test dataset inference on 2h1h1k_1e4_256_30 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 2h1h1k_1e4_256_30 --classifier 2h1h1k  > ../Analysis/logs/avg/2h1h1k_1e4_256_30.txt

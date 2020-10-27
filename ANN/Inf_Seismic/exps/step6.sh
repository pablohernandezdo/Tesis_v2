#!/bin/bash

mkdir -p ../logs/

# Test dataset inference on 2h1k5k_1e3_256_30 model
echo "Running DAS test dataset inference on 2h1k5k_1e3_256_30 model"
python ../inf_test_dataset.py --model_name 2h1k5k_1e3_256_30 --classifier 2h1k5k  > ../logs/2h1k5k_1e3_256_30.txt

# Test dataset inference on 2h5h6k_1e4_256_40 model
echo "Running DAS test dataset inference on 2h5h6k_1e4_256_40 model"
python ../inf_test_dataset.py --model_name 2h5h6k_1e4_256_40 --classifier 2h5h6k  > ../logs/2h5h6k_1e4_256_40.txt

# Test dataset inference on 2h5h6k_1e4_256_35 model
echo "Running DAS test dataset inference on 2h5h6k_1e4_256_35 model"
python ../inf_test_dataset.py --model_name 2h5h6k_1e4_256_35 --classifier 2h5h6k  > ../logs/2h5h6k_1e4_256_35.txt

# Test dataset inference on 2h5h5k_1e4_256_25 model
echo "Running DAS test dataset inference on 2h5h5k_1e4_256_25 model"
python ../inf_test_dataset.py --model_name 2h5h5k_1e4_256_25 --classifier 2h5h5k  > ../logs/2h5h5k_1e4_256_25.txt

# Test dataset inference on 2h5h6k_1e4_256_30 model
echo "Running DAS test dataset inference on 2h5h6k_1e4_256_30 model"
python ../inf_test_dataset.py --model_name 2h5h6k_1e4_256_30 --classifier 2h5h6k  > ../logs/2h5h6k_1e4_256_30.txt

# Test dataset inference on 2h5h5k_1e4_256_35 model
echo "Running DAS test dataset inference on 2h5h6k_1e4_256_25 model"
python ../inf_test_dataset.py --model_name 2h5h6k_1e4_256_25 --classifier 2h5h6k  > ../logs/2h5h6k_1e4_256_25.txt

# Test dataset inference on 2h5h5k_1e4_256_30 model
echo "Running DAS test dataset inference on 2h5h5k_1e4_256_30 model"
python ../inf_test_dataset.py --model_name 2h5h5k_1e4_256_30 --classifier 2h5h5k  > ../logs/2h5h5k_1e4_256_30.txt

# Test dataset inference on 2h5h6k_5e5_256_40 model
echo "Running DAS test dataset inference on 2h5h6k_5e5_256_40 model"
python ../inf_test_dataset.py --model_name 2h5h6k_5e5_256_40 --classifier 2h5h6k  > ../logs/2h5h6k_5e5_256_40.txt

# Test dataset inference on 2h5h5k_1e4_256_35 model
echo "Running DAS test dataset inference on 2h5h5k_1e4_256_35 model"
python ../inf_test_dataset.py --model_name 2h5h5k_1e4_256_35 --classifier 2h5h5k  > ../logs/2h5h5k_1e4_256_35.txt

# Test dataset inference on 2h5h4k_1e4_256_40 model
echo "Running DAS test dataset inference on 2h5h4k_1e4_256_40 model"
python ../inf_test_dataset.py --model_name 2h5h4k_1e4_256_40 --classifier 2h5h4k  > ../logs/2h5h4k_1e4_256_40.txt

#!/bin/bash

mkdir -p ../logs/itd

mf="step4"

# Test dataset inference on 2h1k5k_5e5_256_20 model
echo "Running DAS test dataset inference on 2h1k5k_5e5_256_20 model"
python ../inf_test_dataset.py --model_folder $mf --model_name 2h1k5k_5e5_256_20 --classifier 2h1k5k  > ../logs/itd/2h1k5k_5e5_256_20.txt

# Test dataset inference on 2h1k6k_5e5_256_20 model
echo "Running DAS test dataset inference on 2h1k6k_5e5_256_20 model"
python ../inf_test_dataset.py --model_name $mf 2h1k6k_5e5_256_20 --classifier 2h1k6k  > ../logs/itd/2h1k6k_5e5_256_20.txt

# Test dataset inference on 2h1k6k_5e5_256_30 model
echo "Running DAS test dataset inference on 2h1k6k_5e5_256_30 model"
python ../inf_test_dataset.py --model_name $mf 2h1k6k_5e5_256_30 --classifier 2h1k6k  > ../logs/itd/2h1k6k_5e5_256_30.txt

# Test dataset inference on 2h5h4k_5e5_256_35 model
echo "Running DAS test dataset inference on 2h5h4k_5e5_256_35 model"
python ../inf_test_dataset.py --model_name $mf 2h5h4k_5e5_256_35 --classifier 2h5h4k  > ../logs/itd/2h5h4k_5e5_256_35.txt

# Test dataset inference on 2h1k6k_1e4_256_20 model
echo "Running DAS test dataset inference on 2h1k6k_1e4_256_20 model"
python ../inf_test_dataset.py --model_name $mf 2h1k6k_1e4_256_20 --classifier 2h1k6k  > ../logs/itd/2h1k6k_1e4_256_20.txt

# Test dataset inference on 2h5h6k_5e5_256_25 model
echo "Running DAS test dataset inference on 2h5h6k_1e4_256_25 model"
python ../inf_test_dataset.py --model_name $mf 2h5h6k_5e5_256_25 --classifier 2h5h6k  > ../logs/itd/2h5h6k_5e5_256_25.txt

# Test dataset inference on 2h1k5k_5e5_256_35 model
echo "Running DAS test dataset inference on 2h1k5k_5e5_256_35 model"
python ../inf_test_dataset.py --model_name $mf 2h1k5k_5e5_256_35 --classifier 2h1k5k  > ../logs/itd/2h1k5k_5e5_256_35.txt

# Test dataset inference on 2h1k6k_5e5_256_35 model
echo "Running DAS test dataset inference on 2h1k6k_5e5_256_35 model"
python ../inf_test_dataset.py --model_name $mf 2h1k6k_5e5_256_35 --classifier 2h1k6k  > ../logs/itd/2h1k6k_5e5_256_35.txt

# Test dataset inference on 2h5h5k_1e4_256_20 model
echo "Running DAS test dataset inference on 2h5h5k_1e4_256_20 model"
python ../inf_test_dataset.py --model_name $mf 2h5h5k_1e4_256_20 --classifier 2h5h5k  > ../logs/itd/2h5h5k_1e4_256_20.txt

# Test dataset inference on 2h5h2k_5e5_256_30 model
echo "Running DAS test dataset inference on 2h5h2k_5e5_256_30 model"
python ../inf_test_dataset.py --model_name $mf 2h5h2k_5e5_256_30 --classifier 2h5h4k  > ../logs/itd/2h5h2k_5e5_256_30.txt

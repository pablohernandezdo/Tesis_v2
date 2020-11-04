#!/bin/bash

echo "Evaluating model 2h1k5k_5e5_256"
python ../eval_curves.py --test_path $tst \
              --classifier 2h1k5k --model_name 2h1k5k_5e5_256_40 > ../logs/eval/2h1k5k_5e5_256_40.txt

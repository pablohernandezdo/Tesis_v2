#!/bin/bash

echo "Creating summary of reports excel file"
python ../reports2excel.py --folder_name 'eval' --xls_name 'step1CNN_curves' --n_thresh 19

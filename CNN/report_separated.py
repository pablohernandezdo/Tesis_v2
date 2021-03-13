import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder', default='Analysis/CSVOutputs',
                        help='Path to CSV files folder')
    parser.add_argument('--xls_name',
                        help='Best models excel file name')
    parser.add_argument("--beta", type=float, default=2,
                        help="Fscore beta parameter")
    args = parser.parse_args()




if __name__ == "__main__":
    main()

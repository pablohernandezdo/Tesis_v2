import os
import argparse

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", default='default', help="Folder to save model")
    parser.add_argument('--model_name', default='default', help='Model name')
    parser.add_argument('--n_seis', type=int, default=8923, help='Number of examples in each dataset category')
    parser.add_argument('--n_nseis', type=int, default=11022, help='Number of examples in each dataset category')
    parser.add_argument('--n_bins', type=int, default=100, help='Number of histogram bins')
    args = parser.parse_args()

    Path(f"../Analysis/Output_values/{args.model_folder}").mkdir(parents=True, exist_ok=True)

    output_file = os.path.join('../Analysis/Output_values/', args.model_folder, 'outputs_' + args.model_name + '.txt')

    seismic_outputs = []
    nseismic_outputs = []

    with open(output_file, 'r') as f:
        # Seismic initial line
        f.readline()

        for i in range(args.n_seis):
            seismic_outputs.append(float(f.readline().strip()))

        # Non Seismic initial line
        f.readline()

        for i in range(args.n_nseis):
            nseismic_outputs.append(float(f.readline().strip()))

    # Plot histogram
    plot_histogram(seismic_outputs, nseismic_outputs, args.model_folder, args.model_name, args.n_bins)


def plot_histogram(s_outputs, ns_outputs, model_folder, model_name, n_bins):

    plt.figure()

    n_seis, bins_seis, patches_seis = plt.hist(s_outputs, bins=n_bins, color='blue', alpha=0.6, label='Seismic')
    n_nseis, bins_nseis, patches_nseis = plt.hist(ns_outputs, bins=n_bins, color='red', alpha=0.6, label='Non seismic')

    plt.title(f'Output values histogram model {model_name}')
    plt.xlabel('Net output value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(f'../Analysis/Histograms/{model_folder}/Histogram_{model_name}.png')


if __name__ == "__main__":
    main()

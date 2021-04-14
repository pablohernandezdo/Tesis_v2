import time
import argparse
import numpy as np
from pathlib import Path
from humanfriendly import format_timespan

import tqdm
import torch
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, lfilter, detrend

from torch.utils.data import DataLoader

from model import *
from dataset import HDF5Dataset


def main():
    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='1h6k_test_model',
                        help="Name of model to eval")
    parser.add_argument("--model_folder", default='test',
                        help="Model to eval folder")
    parser.add_argument("--classifier", default='1h6k',
                        help="Choose classifier architecture")
    parser.add_argument("--francia_path",
                        default='../../Data/Francia/'
                                'Earthquake_1p9_Var_BP_2p5_15Hz.mat',
                        help="HDF5 test Dataset path")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Size of the training batches")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Test dataset
    traces = get_francia(args.francia_path)

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Count number of parameters
    params = count_parameters(net)

    # Load from trained model
    net.load_state_dict(torch.load(f'../models/{args.model_folder}/'
                                   f'{args.model_name}.pth'))
    net.eval()

    # EVAL #
    # List of outputs and labels used to create pd dataframe
    out_vals = []

    with tqdm.tqdm(total=len(traces),
                   desc=f'Francia dataset evaluation') as data_bar:

        with torch.no_grad():
            for data in traces:

                data = torch.from_numpy(data).float()
                data = data.to(device)
                outputs = net(data)

                out_vals.append(outputs.item())
                data_bar.update(1)

    with open(f'../Results/francia_results.txt', 'w') as f:
        f.write("\n".join(map(str, out_vals)))

    eval_end = time.time()
    total_time = eval_end - start_time

    print(f'Total execution time: {format_timespan(total_time)}\n'
          f'Number of network parameters: {params}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_francia(dataset_path):
    # Read file and load data
    f = sio.loadmat(dataset_path)
    traces = f["StrainFilt"]
    fs = 100

    traces = preprocess(traces, fs)

    return traces


def preprocess(traces, fs):

    new_traces = []

    for trace in traces:
        # Filter 50 Hz
        trace = butter_lowpasspass_filter(trace, 49, fs)

        # Detrending
        trace = detrend(trace)

        # Media cero
        trace = trace - np.mean(trace)
        new_traces.append(trace)

    return np.asarray(new_traces)


def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, output='ba')
    return b, a


def butter_lowpasspass_filter(dat, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, dat)
    return y


def get_classifier(x):
    if x == 'Cnn1_6k':
        return Cnn1_6k()
    if x == 'Cnn1_5k':
        return Cnn1_5k()
    if x == 'Cnn1_4k':
        return Cnn1_4k()
    if x == 'Cnn1_3k':
        return Cnn1_3k()
    if x == 'Cnn1_2k':
        return Cnn1_2k()
    if x == 'Cnn1_1k':
        return Cnn1_1k()
    if x == 'Cnn1_5h':
        return Cnn1_5h()
    if x == 'Cnn1_2h':
        return Cnn1_2h()
    if x == 'Cnn1_5h':
        return Cnn1_1h()
    if x == 'Cnn1_10':
        return Cnn1_10()
    if x == 'Cnn1_6k_6k':
        return Cnn1_6k_6k()
    if x == 'Cnn1_6k_5k':
        return Cnn1_6k_5k()
    if x == 'Cnn1_6k_4k':
        return Cnn1_6k_4k()
    if x == 'Cnn1_6k_3k':
        return Cnn1_6k_3k()
    if x == 'Cnn1_6k_2k':
        return Cnn1_6k_2k()
    if x == 'Cnn1_6k_1k':
        return Cnn1_6k_1k()
    if x == 'Cnn1_6k_5h':
        return Cnn1_6k_5h()
    if x == 'Cnn1_6k_2h':
        return Cnn1_6k_2h()
    if x == 'Cnn1_6k_1h':
        return Cnn1_6k_1h()
    if x == 'Cnn1_6k_10':
        return Cnn1_6k_10()
    if x == 'Cnn1_5k_6k':
        return Cnn1_5k_6k()
    if x == 'Cnn1_5k_5k':
        return Cnn1_5k_5k()
    if x == 'Cnn1_5k_4k':
        return Cnn1_5k_4k()
    if x == 'Cnn1_5k_3k':
        return Cnn1_5k_3k()
    if x == 'Cnn1_5k_2k':
        return Cnn1_5k_2k()
    if x == 'Cnn1_5k_1k':
        return Cnn1_5k_1k()
    if x == 'Cnn1_5k_5h':
        return Cnn1_5k_5h()
    if x == 'Cnn1_5k_2h':
        return Cnn1_5k_2h()
    if x == 'Cnn1_5k_1h':
        return Cnn1_5k_1h()
    if x == 'Cnn1_5k_10':
        return Cnn1_5k_10()
    if x == 'Cnn1_4k_6k':
        return Cnn1_4k_6k()
    if x == 'Cnn1_4k_5k':
        return Cnn1_4k_5k()
    if x == 'Cnn1_4k_4k':
        return Cnn1_4k_4k()
    if x == 'Cnn1_4k_3k':
        return Cnn1_4k_3k()
    if x == 'Cnn1_4k_2k':
        return Cnn1_4k_2k()
    if x == 'Cnn1_4k_1k':
        return Cnn1_4k_1k()
    if x == 'Cnn1_4k_5h':
        return Cnn1_4k_5h()
    if x == 'Cnn1_4k_2h':
        return Cnn1_4k_2h()
    if x == 'Cnn1_4k_1h':
        return Cnn1_4k_1h()
    if x == 'Cnn1_4k_10':
        return Cnn1_4k_10()
    if x == 'Cnn1_3k_6k':
        return Cnn1_3k_6k()
    if x == 'Cnn1_3k_5k':
        return Cnn1_3k_5k()
    if x == 'Cnn1_3k_4k':
        return Cnn1_3k_4k()
    if x == 'Cnn1_3k_3k':
        return Cnn1_3k_3k()
    if x == 'Cnn1_3k_2k':
        return Cnn1_3k_2k()
    if x == 'Cnn1_3k_1k':
        return Cnn1_3k_1k()
    if x == 'Cnn1_3k_5h':
        return Cnn1_3k_5h()
    if x == 'Cnn1_3k_2h':
        return Cnn1_3k_2h()
    if x == 'Cnn1_3k_1h':
        return Cnn1_3k_1h()
    if x == 'Cnn1_3k_10':
        return Cnn1_3k_10()
    if x == 'Cnn1_2k_6k':
        return Cnn1_2k_6k()
    if x == 'Cnn1_2k_5k':
        return Cnn1_2k_5k()
    if x == 'Cnn1_2k_4k':
        return Cnn1_2k_4k()
    if x == 'Cnn1_2k_3k':
        return Cnn1_2k_3k()
    if x == 'Cnn1_2k_2k':
        return Cnn1_2k_2k()
    if x == 'Cnn1_2k_1k':
        return Cnn1_2k_1k()
    if x == 'Cnn1_2k_5h':
        return Cnn1_2k_5h()
    if x == 'Cnn1_2k_2h':
        return Cnn1_2k_2h()
    if x == 'Cnn1_2k_1h':
        return Cnn1_2k_1h()
    if x == 'Cnn1_2k_10':
        return Cnn1_2k_10()
    if x == 'Cnn1_1k_6k':
        return Cnn1_1k_6k()
    if x == 'Cnn1_1k_5k':
        return Cnn1_1k_5k()
    if x == 'Cnn1_1k_4k':
        return Cnn1_1k_4k()
    if x == 'Cnn1_1k_3k':
        return Cnn1_1k_3k()
    if x == 'Cnn1_1k_2k':
        return Cnn1_1k_2k()
    if x == 'Cnn1_1k_1k':
        return Cnn1_1k_1k()
    if x == 'Cnn1_1k_5h':
        return Cnn1_1k_5h()
    if x == 'Cnn1_1k_2h':
        return Cnn1_1k_2h()
    if x == 'Cnn1_1k_1h':
        return Cnn1_1k_1h()
    if x == 'Cnn1_1k_10':
        return Cnn1_1k_10()
    if x == 'Cnn1_5h_6k':
        return Cnn1_5h_6k()
    if x == 'Cnn1_5h_5k':
        return Cnn1_5h_5k()
    if x == 'Cnn1_5h_4k':
        return Cnn1_5h_4k()
    if x == 'Cnn1_5h_3k':
        return Cnn1_5h_3k()
    if x == 'Cnn1_5h_2k':
        return Cnn1_5h_2k()
    if x == 'Cnn1_5h_1k':
        return Cnn1_5h_1k()
    if x == 'Cnn1_5h_5h':
        return Cnn1_5h_5h()
    if x == 'Cnn1_5h_2h':
        return Cnn1_5h_2h()
    if x == 'Cnn1_5h_1h':
        return Cnn1_5h_1h()
    if x == 'Cnn1_5h_10':
        return Cnn1_5h_10()
    if x == 'Cnn1_2h_6k':
        return Cnn1_2h_6k()
    if x == 'Cnn1_2h_5k':
        return Cnn1_2h_5k()
    if x == 'Cnn1_2h_4k':
        return Cnn1_2h_4k()
    if x == 'Cnn1_2h_3k':
        return Cnn1_2h_3k()
    if x == 'Cnn1_2h_2k':
        return Cnn1_2h_2k()
    if x == 'Cnn1_2h_1k':
        return Cnn1_2h_1k()
    if x == 'Cnn1_2h_5h':
        return Cnn1_2h_5h()
    if x == 'Cnn1_2h_2h':
        return Cnn1_2h_2h()
    if x == 'Cnn1_2h_1h':
        return Cnn1_2h_1h()
    if x == 'Cnn1_2h_10':
        return Cnn1_2h_10()
    if x == 'Cnn1_1h_6k':
        return Cnn1_1h_6k()
    if x == 'Cnn1_1h_5k':
        return Cnn1_1h_5k()
    if x == 'Cnn1_1h_4k':
        return Cnn1_1h_4k()
    if x == 'Cnn1_1h_3k':
        return Cnn1_1h_3k()
    if x == 'Cnn1_1h_2k':
        return Cnn1_1h_2k()
    if x == 'Cnn1_1h_1k':
        return Cnn1_1h_1k()
    if x == 'Cnn1_1h_5h':
        return Cnn1_1h_5h()
    if x == 'Cnn1_1h_2h':
        return Cnn1_1h_2h()
    if x == 'Cnn1_1h_1h':
        return Cnn1_1h_1h()
    if x == 'Cnn1_1h_10':
        return Cnn1_1h_10()
    if x == 'Cnn1_10_6k':
        return Cnn1_10_6k()
    if x == 'Cnn1_10_5k':
        return Cnn1_10_5k()
    if x == 'Cnn1_10_4k':
        return Cnn1_10_4k()
    if x == 'Cnn1_10_3k':
        return Cnn1_10_3k()
    if x == 'Cnn1_10_2k':
        return Cnn1_10_2k()
    if x == 'Cnn1_10_1k':
        return Cnn1_10_1k()
    if x == 'Cnn1_10_5h':
        return Cnn1_10_5h()
    if x == 'Cnn1_10_2h':
        return Cnn1_10_2h()
    if x == 'Cnn1_10_1h':
        return Cnn1_10_1h()
    if x == 'Cnn1_10_10':
        return Cnn1_10_10()
    if x == 'Cnn2_6k':
        return Cnn2_6k()
    if x == 'Cnn2_5k':
        return Cnn2_5k()
    if x == 'Cnn2_4k':
        return Cnn2_4k()
    if x == 'Cnn2_3k':
        return Cnn2_3k()
    if x == 'Cnn2_2k':
        return Cnn2_2k()
    if x == 'Cnn2_1k':
        return Cnn2_1k()
    if x == 'Cnn2_5h':
        return Cnn2_5h()
    if x == 'Cnn2_2h':
        return Cnn2_2h()
    if x == 'Cnn2_5h':
        return Cnn2_1h()
    if x == 'Cnn2_10':
        return Cnn2_10()
    if x == 'Cnn2_6k_6k':
        return Cnn2_6k_6k()
    if x == 'Cnn2_6k_5k':
        return Cnn2_6k_5k()
    if x == 'Cnn2_6k_4k':
        return Cnn2_6k_4k()
    if x == 'Cnn2_6k_3k':
        return Cnn2_6k_3k()
    if x == 'Cnn2_6k_2k':
        return Cnn2_6k_2k()
    if x == 'Cnn2_6k_1k':
        return Cnn2_6k_1k()
    if x == 'Cnn2_6k_5h':
        return Cnn2_6k_5h()
    if x == 'Cnn2_6k_2h':
        return Cnn2_6k_2h()
    if x == 'Cnn2_6k_1h':
        return Cnn2_6k_1h()
    if x == 'Cnn2_6k_10':
        return Cnn2_6k_10()
    if x == 'Cnn2_5k_6k':
        return Cnn2_5k_6k()
    if x == 'Cnn2_5k_5k':
        return Cnn2_5k_5k()
    if x == 'Cnn2_5k_4k':
        return Cnn2_5k_4k()
    if x == 'Cnn2_5k_3k':
        return Cnn2_5k_3k()
    if x == 'Cnn2_5k_2k':
        return Cnn2_5k_2k()
    if x == 'Cnn2_5k_1k':
        return Cnn2_5k_1k()
    if x == 'Cnn2_5k_5h':
        return Cnn2_5k_5h()
    if x == 'Cnn2_5k_2h':
        return Cnn2_5k_2h()
    if x == 'Cnn2_5k_1h':
        return Cnn2_5k_1h()
    if x == 'Cnn2_5k_10':
        return Cnn2_5k_10()
    if x == 'Cnn2_4k_6k':
        return Cnn2_4k_6k()
    if x == 'Cnn2_4k_5k':
        return Cnn2_4k_5k()
    if x == 'Cnn2_4k_4k':
        return Cnn2_4k_4k()
    if x == 'Cnn2_4k_3k':
        return Cnn2_4k_3k()
    if x == 'Cnn2_4k_2k':
        return Cnn2_4k_2k()
    if x == 'Cnn2_4k_1k':
        return Cnn2_4k_1k()
    if x == 'Cnn2_4k_5h':
        return Cnn2_4k_5h()
    if x == 'Cnn2_4k_2h':
        return Cnn2_4k_2h()
    if x == 'Cnn2_4k_1h':
        return Cnn2_4k_1h()
    if x == 'Cnn2_4k_10':
        return Cnn2_4k_10()
    if x == 'Cnn2_3k_6k':
        return Cnn2_3k_6k()
    if x == 'Cnn2_3k_5k':
        return Cnn2_3k_5k()
    if x == 'Cnn2_3k_4k':
        return Cnn2_3k_4k()
    if x == 'Cnn2_3k_3k':
        return Cnn2_3k_3k()
    if x == 'Cnn2_3k_2k':
        return Cnn2_3k_2k()
    if x == 'Cnn2_3k_1k':
        return Cnn2_3k_1k()
    if x == 'Cnn2_3k_5h':
        return Cnn2_3k_5h()
    if x == 'Cnn2_3k_2h':
        return Cnn2_3k_2h()
    if x == 'Cnn2_3k_1h':
        return Cnn2_3k_1h()
    if x == 'Cnn2_3k_10':
        return Cnn2_3k_10()
    if x == 'Cnn2_2k_6k':
        return Cnn2_2k_6k()
    if x == 'Cnn2_2k_5k':
        return Cnn2_2k_5k()
    if x == 'Cnn2_2k_4k':
        return Cnn2_2k_4k()
    if x == 'Cnn2_2k_3k':
        return Cnn2_2k_3k()
    if x == 'Cnn2_2k_2k':
        return Cnn2_2k_2k()
    if x == 'Cnn2_2k_1k':
        return Cnn2_2k_1k()
    if x == 'Cnn2_2k_5h':
        return Cnn2_2k_5h()
    if x == 'Cnn2_2k_2h':
        return Cnn2_2k_2h()
    if x == 'Cnn2_2k_1h':
        return Cnn2_2k_1h()
    if x == 'Cnn2_2k_10':
        return Cnn2_2k_10()
    if x == 'Cnn2_1k_6k':
        return Cnn2_1k_6k()
    if x == 'Cnn2_1k_5k':
        return Cnn2_1k_5k()
    if x == 'Cnn2_1k_4k':
        return Cnn2_1k_4k()
    if x == 'Cnn2_1k_3k':
        return Cnn2_1k_3k()
    if x == 'Cnn2_1k_2k':
        return Cnn2_1k_2k()
    if x == 'Cnn2_1k_1k':
        return Cnn2_1k_1k()
    if x == 'Cnn2_1k_5h':
        return Cnn2_1k_5h()
    if x == 'Cnn2_1k_2h':
        return Cnn2_1k_2h()
    if x == 'Cnn2_1k_1h':
        return Cnn2_1k_1h()
    if x == 'Cnn2_1k_10':
        return Cnn2_1k_10()
    if x == 'Cnn2_5h_6k':
        return Cnn2_5h_6k()
    if x == 'Cnn2_5h_5k':
        return Cnn2_5h_5k()
    if x == 'Cnn2_5h_4k':
        return Cnn2_5h_4k()
    if x == 'Cnn2_5h_3k':
        return Cnn2_5h_3k()
    if x == 'Cnn2_5h_2k':
        return Cnn2_5h_2k()
    if x == 'Cnn2_5h_1k':
        return Cnn2_5h_1k()
    if x == 'Cnn2_5h_5h':
        return Cnn2_5h_5h()
    if x == 'Cnn2_5h_2h':
        return Cnn2_5h_2h()
    if x == 'Cnn2_5h_1h':
        return Cnn2_5h_1h()
    if x == 'Cnn2_5h_10':
        return Cnn2_5h_10()
    if x == 'Cnn2_2h_6k':
        return Cnn2_2h_6k()
    if x == 'Cnn2_2h_5k':
        return Cnn2_2h_5k()
    if x == 'Cnn2_2h_4k':
        return Cnn2_2h_4k()
    if x == 'Cnn2_2h_3k':
        return Cnn2_2h_3k()
    if x == 'Cnn2_2h_2k':
        return Cnn2_2h_2k()
    if x == 'Cnn2_2h_1k':
        return Cnn2_2h_1k()
    if x == 'Cnn2_2h_5h':
        return Cnn2_2h_5h()
    if x == 'Cnn2_2h_2h':
        return Cnn2_2h_2h()
    if x == 'Cnn2_2h_1h':
        return Cnn2_2h_1h()
    if x == 'Cnn2_2h_10':
        return Cnn2_2h_10()
    if x == 'Cnn2_1h_6k':
        return Cnn2_1h_6k()
    if x == 'Cnn2_1h_5k':
        return Cnn2_1h_5k()
    if x == 'Cnn2_1h_4k':
        return Cnn2_1h_4k()
    if x == 'Cnn2_1h_3k':
        return Cnn2_1h_3k()
    if x == 'Cnn2_1h_2k':
        return Cnn2_1h_2k()
    if x == 'Cnn2_1h_1k':
        return Cnn2_1h_1k()
    if x == 'Cnn2_1h_5h':
        return Cnn2_1h_5h()
    if x == 'Cnn2_1h_2h':
        return Cnn2_1h_2h()
    if x == 'Cnn2_1h_1h':
        return Cnn2_1h_1h()
    if x == 'Cnn2_1h_10':
        return Cnn2_1h_10()
    if x == 'Cnn2_10_6k':
        return Cnn2_10_6k()
    if x == 'Cnn2_10_5k':
        return Cnn2_10_5k()
    if x == 'Cnn2_10_4k':
        return Cnn2_10_4k()
    if x == 'Cnn2_10_3k':
        return Cnn2_10_3k()
    if x == 'Cnn2_10_2k':
        return Cnn2_10_2k()
    if x == 'Cnn2_10_1k':
        return Cnn2_10_1k()
    if x == 'Cnn2_10_5h':
        return Cnn2_10_5h()
    if x == 'Cnn2_10_2h':
        return Cnn2_10_2h()
    if x == 'Cnn2_10_1h':
        return Cnn2_10_1h()
    if x == 'Cnn2_10_10':
        return Cnn2_10_10()
    else:
        return Cnn2_10_10()


if __name__ == "__main__":
    main()

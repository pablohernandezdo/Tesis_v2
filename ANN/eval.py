import time
import argparse
from pathlib import Path

import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader
from humanfriendly import format_timespan

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
                        help="Folder to save model")
    parser.add_argument("--classifier", default='1h6k',
                        help="Choose classifier architecture")
    parser.add_argument("--train_path", default='Train_data.hdf5',
                        help="HDF5 train Dataset path")
    parser.add_argument("--test_path", default='Test_data.hdf5',
                        help="HDF5 test Dataset path")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Size of the training batches")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train dataset
    train_set = HDF5Dataset(args.train_path)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size, shuffle=True)

    # Test dataset
    test_set = HDF5Dataset(args.test_path)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size, shuffle=True)

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Count number of parameters
    params = count_parameters(net)

    # Load from trained model
    net.load_state_dict(torch.load(f'../models/{args.model_folder}/'
                                   f'{args.model_name}.pth'))
    net.eval()

    # Evaluate model on training dataset

    evaluate_dataset(train_loader, 'Train', device,
                     net, args.model_name, args.model_folder,
                     '../Analysis/CSVOutputs')

    train_end = time.time()
    train_time = train_end - start_time

    # Evaluate model on test set
    evaluate_dataset(test_loader, 'Test', device,
                     net, args.model_name, args.model_folder,
                     '../Analysis/CSVOutputs')

    eval_end = time.time()
    eval_time = eval_end - train_end
    total_time = eval_end - start_time

    print(f'Training evaluation time: {format_timespan(train_time)}\n'
          f'Test evaluation time: {format_timespan(eval_time)}\n'
          f'Total execution time: {format_timespan(total_time)}\n\n'
          f'Number of network parameters: {params}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_dataset(data_loader, dataset_name, device, net,
                     model_name, model_folder, csv_folder):

    # List of outputs and labels used to create pd dataframe
    dataframe_rows_list = []

    with tqdm.tqdm(total=len(data_loader),
                   desc=f'{dataset_name} dataset evaluation') as data_bar:

        with torch.no_grad():
            for data in data_loader:

                traces, labels = data[0].to(device), data[1].to(device)
                outputs = net(traces)

                for out, lab in zip(outputs, labels):
                    new_row = {'out': out.item(), 'label': lab.item()}
                    dataframe_rows_list.append(new_row)

                data_bar.update(1)

    test_outputs = pd.DataFrame(dataframe_rows_list)

    # Create csv folder if necessary
    save_folder = f'{csv_folder}/{dataset_name}/{model_folder}'
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    # Save outputs and labels to csv file
    test_outputs.to_csv(f'{save_folder}/{model_name}.csv', index=False)


def get_classifier(x):
    return {
        '1h6k': OneHidden6k(),
        '1h5k': OneHidden5k(),
        '1h4k': OneHidden4k(),
        '1h3k': OneHidden3k(),
        '1h2k': OneHidden2k(),
        '1h1k': OneHidden1k(),
        '1h5h': OneHidden5h(),
        '1h1h': OneHidden1h(),
        '1h10': OneHidden10(),
        '1h1': OneHidden1(),
        '2h6k6k': TwoHidden6k6k(),
        '2h6k5k': TwoHidden6k5k(),
        '2h6k4k': TwoHidden6k4k(),
        '2h6k3k': TwoHidden6k3k(),
        '2h6k2k': TwoHidden6k2k(),
        '2h6k1k': TwoHidden6k1k(),
        '2h6k5h': TwoHidden6k5h(),
        '2h6k1h': TwoHidden6k1h(),
        '2h6k10': TwoHidden6k10(),
        '2h6k1': TwoHidden6k1(),
        '2h5k6k': TwoHidden5k6k(),
        '2h5k5k': TwoHidden5k5k(),
        '2h5k4k': TwoHidden5k4k(),
        '2h5k3k': TwoHidden5k3k(),
        '2h5k2k': TwoHidden5k2k(),
        '2h5k1k': TwoHidden5k1k(),
        '2h5k5h': TwoHidden5k5h(),
        '2h5k1h': TwoHidden5k1h(),
        '2h5k10': TwoHidden5k10(),
        '2h5k1': TwoHidden5k1(),
        '2h4k6k': TwoHidden4k6k(),
        '2h4k5k': TwoHidden4k5k(),
        '2h4k4k': TwoHidden4k4k(),
        '2h4k3k': TwoHidden4k3k(),
        '2h4k2k': TwoHidden4k2k(),
        '2h4k1k': TwoHidden4k1k(),
        '2h4k5h': TwoHidden4k5h(),
        '2h4k1h': TwoHidden4k1h(),
        '2h4k10': TwoHidden4k10(),
        '2h4k1': TwoHidden4k1(),
        '2h3k6k': TwoHidden3k6k(),
        '2h3k5k': TwoHidden3k5k(),
        '2h3k4k': TwoHidden3k4k(),
        '2h3k3k': TwoHidden3k3k(),
        '2h3k2k': TwoHidden3k2k(),
        '2h3k1k': TwoHidden3k1k(),
        '2h3k5h': TwoHidden3k5h(),
        '2h3k1h': TwoHidden3k1h(),
        '2h3k10': TwoHidden3k10(),
        '2h3k1': TwoHidden3k1(),
        '2h2k6k': TwoHidden2k6k(),
        '2h2k5k': TwoHidden2k5k(),
        '2h2k4k': TwoHidden2k4k(),
        '2h2k3k': TwoHidden2k3k(),
        '2h2k2k': TwoHidden2k2k(),
        '2h2k1k': TwoHidden2k1k(),
        '2h2k5h': TwoHidden2k5h(),
        '2h2k1h': TwoHidden2k1h(),
        '2h2k10': TwoHidden2k10(),
        '2h2k1': TwoHidden2k1(),
        '2h1k6k': TwoHidden1k6k(),
        '2h1k5k': TwoHidden1k5k(),
        '2h1k4k': TwoHidden1k4k(),
        '2h1k3k': TwoHidden1k3k(),
        '2h1k2k': TwoHidden1k2k(),
        '2h1k1k': TwoHidden1k1k(),
        '2h1k5h': TwoHidden1k5h(),
        '2h1k1h': TwoHidden1k1h(),
        '2h1k10': TwoHidden1k10(),
        '2h1k1': TwoHidden1k1(),
        '2h5h6k': TwoHidden5h6k(),
        '2h5h5k': TwoHidden5h5k(),
        '2h5h4k': TwoHidden5h4k(),
        '2h5h3k': TwoHidden5h3k(),
        '2h5h2k': TwoHidden5h2k(),
        '2h5h1k': TwoHidden5h1k(),
        '2h5h5h': TwoHidden5h5h(),
        '2h5h1h': TwoHidden5h1h(),
        '2h5h10': TwoHidden5h10(),
        '2h5h1': TwoHidden5h1(),
        '2h1h6k': TwoHidden1h6k(),
        '2h1h5k': TwoHidden1h5k(),
        '2h1h4k': TwoHidden1h4k(),
        '2h1h3k': TwoHidden1h3k(),
        '2h1h2k': TwoHidden1h2k(),
        '2h1h1k': TwoHidden1h1k(),
        '2h1h5h': TwoHidden1h5h(),
        '2h1h1h': TwoHidden1h1h(),
        '2h1h10': TwoHidden1h10(),
        '2h1h1': TwoHidden1h1(),
        '2h10_6k': TwoHidden10_6k(),
        '2h10_5k': TwoHidden10_5k(),
        '2h10_4k': TwoHidden10_4k(),
        '2h10_3k': TwoHidden10_3k(),
        '2h10_2k': TwoHidden10_2k(),
        '2h10_1k': TwoHidden10_1k(),
        '2h10_5h': TwoHidden10_5h(),
        '2h10_1h': TwoHidden10_1h(),
        '2h10_10': TwoHidden10_10(),
        '2h10_1': TwoHidden1_1(),
        '2h1_6k': TwoHidden1_6k(),
        '2h1_5k': TwoHidden1_5k(),
        '2h1_4k': TwoHidden1_4k(),
        '2h1_3k': TwoHidden1_3k(),
        '2h1_2k': TwoHidden1_2k(),
        '2h1_1k': TwoHidden1_1k(),
        '2h1_5h': TwoHidden1_5h(),
        '2h1_1h': TwoHidden1_1h(),
        '2h1_10': TwoHidden1_10(),
        '2h1_1': TwoHidden1_1(),
    }.get(x, OneHidden6k())


if __name__ == "__main__":
    main()

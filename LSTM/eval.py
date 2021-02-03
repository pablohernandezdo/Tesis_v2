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
    parser.add_argument("--model_name", default='XXL_lr0000001_bs32', help="Name of model to eval")
    parser.add_argument("--model_folder", default='default', help="Folder to save model")
    parser.add_argument("--classifier", default='XXL', help="Choose classifier architecture, C, S, XS, XL, XXL, XXXL")
    parser.add_argument("--train_path", default='Train_data.hdf5', help="HDF5 train Dataset path")
    parser.add_argument("--test_path", default='Test_data.hdf5', help="HDF5 test Dataset path")
    parser.add_argument("--batch_size", type=int, default=256, help="Size of the batches")
    parser.add_argument("--beta", type=float, default=2, help="Fscore beta parameter")
    args = parser.parse_args()

    # Create csv files folder
    Path(f"../Analysis/OutputsCSV/{args.model_folder}/train").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/OutputsCSV/{args.model_folder}/eval").mkdir(parents=True, exist_ok=True)

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train dataset
    train_dataset = HDF5Dataset(args.train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Test dataset
    test_dataset = HDF5Dataset(args.test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Count number of parameters
    nparams = count_parameters(net)

    # Load from trained model
    net.load_state_dict(torch.load('../models/' + args.model_folder + '/' + args.model_name + '.pth'))
    net.eval()

    # Evaluate model on training dataset

    train_rows_list = []
    with tqdm.tqdm(total=len(train_loader), desc='Train dataset evaluation', position=0) as train_bar:
        with torch.no_grad():
            for data in train_loader:
                traces, labels = data[0].to(device), data[1].to(device)
                outputs = net(traces)

                for out, lab in zip(outputs, labels):
                    new_row = {'out': out.item(), 'label': lab.item()}
                    train_rows_list.append(new_row)

                train_bar.update(1)

    train_outputs = pd.DataFrame(train_rows_list)
    train_outputs.to_csv(f'../Analysis/OutputsCSV/{args.model_folder}/train/{args.model_name}.csv', index=False)

    eval_1 = time.time()
    ev_1 = eval_1 - start_time

    # Evaluate model on test set

    test_rows_list = []
    with tqdm.tqdm(total=len(test_loader), desc='Test dataset evaluation', position=0) as test_bar:
        with torch.no_grad():
            for data in test_loader:
                traces, labels = data[0].to(device), data[1].to(device)
                outputs = net(traces)

                for out, lab in zip(outputs, labels):
                    new_row = {'out': out.item(), 'label': lab.item()}
                    test_rows_list.append(new_row)

                test_bar.update(1)

    test_outputs = pd.DataFrame(test_rows_list)
    test_outputs.to_csv(f'../Analysis/OutputsCSV/{args.model_folder}/eval/{args.model_name}.csv', index=False)

    eval_2 = time.time()
    ev_2 = eval_2 - eval_1
    ev_t = eval_2 - start_time

    print(f'Training evaluation time: {format_timespan(ev_1)}\n'
          f'Test evaluation time: {format_timespan(ev_2)}\n'
          f'Total execution time: {format_timespan(ev_t)}\n\n'
          f'Number of network parameters: {nparams}')


def print_metrics(tp, fp, tn, fn, beta):
    acc = (tp + tn) / (tp + fp + tn + fn)

    # Evaluation metrics
    if (not tp) and (not fp):
        precision = 1
    else:
        precision = tp / (tp + fp)

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    if (not precision) and (not recall):
        fscore = 0
    else:
        fscore = (1 + beta**2) * (precision * recall) / ((beta**2)*precision + recall)

    # Results
    print(f'Total seismic traces: {tp + fn}\n'
          f'Total non seismic traces: {tn + fp}\n'
          f'correct: {tp + tn} / {tp + fp + tn + fn} \n\n'
          f'True positives: {tp}\n'
          f'True negatives: {tn}\n'
          f'False positives: {fp}\n'
          f'False negatives: {fn}\n\n'
          f'Accuracy: {acc:5.3f}\n'
          f'Precision: {precision:5.3f}\n'
          f'Recall: {recall:5.3f}\n'
          f'False positive rate: {fpr:5.3f}\n'
          f'F-score: {fscore:5.3f}\n')

    return precision, recall, fpr, fscore


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_classifier(x):
    if x == 'Lstm_16_16_1_1':
        return Lstm_16_16_1_1()
    if x == 'Lstm_16_16_2_1':
        return Lstm_16_16_2_1()
    if x == 'Lstm_16_16_5_1':
        return Lstm_16_16_5_1()
    if x == 'Lstm_16_16_10_1':
        return Lstm_16_16_10_1()
    if x == 'Lstm_16_16_20_1':
        return Lstm_16_16_20_1()
    if x == 'Lstm_16_32_1_1':
        return Lstm_16_32_1_1()
    if x == 'Lstm_16_32_2_1':
        return Lstm_16_32_2_1()
    if x == 'Lstm_16_32_5_1':
        return Lstm_16_32_5_1()
    if x == 'Lstm_16_32_10_1':
        return Lstm_16_32_10_1()
    if x == 'Lstm_16_32_20_1':
        return Lstm_16_32_20_1()
    if x == 'Lstm_16_64_1_1':
        return Lstm_16_64_1_1()
    if x == 'Lstm_16_64_2_1':
        return Lstm_16_64_2_1()
    if x == 'Lstm_16_64_5_1':
        return Lstm_16_64_5_1()
    if x == 'Lstm_16_64_10_1':
        return Lstm_16_64_10_1()
    if x == 'Lstm_16_64_20_1':
        return Lstm_16_64_20_1()
    if x == 'Lstm_16_128_1_1':
        return Lstm_16_128_1_1()
    if x == 'Lstm_16_128_2_1':
        return Lstm_16_128_2_1()
    if x == 'Lstm_16_128_5_1':
        return Lstm_16_128_5_1()
    if x == 'Lstm_16_128_10_1':
        return Lstm_16_128_10_1()
    if x == 'Lstm_16_128_20_1':
        return Lstm_16_128_20_1()
    if x == 'Lstm_16_256_1_1':
        return Lstm_16_256_1_1()
    if x == 'Lstm_16_256_2_1':
        return Lstm_16_256_2_1()
    if x == 'Lstm_16_256_5_1':
        return Lstm_16_256_5_1()
    if x == 'Lstm_16_256_10_1':
        return Lstm_16_256_10_1()
    if x == 'Lstm_16_256_20_1':
        return Lstm_16_256_20_1()
    if x == 'Lstm_32_16_1_1':
        return Lstm_32_16_1_1()
    if x == 'Lstm_32_16_2_1':
        return Lstm_32_16_2_1()
    if x == 'Lstm_32_16_5_1':
        return Lstm_32_16_5_1()
    if x == 'Lstm_32_16_10_1':
        return Lstm_32_16_10_1()
    if x == 'Lstm_32_16_20_1':
        return Lstm_32_16_20_1()
    if x == 'Lstm_32_32_1_1':
        return Lstm_32_32_1_1()
    if x == 'Lstm_32_32_2_1':
        return Lstm_32_32_2_1()
    if x == 'Lstm_32_32_5_1':
        return Lstm_32_32_5_1()
    if x == 'Lstm_32_32_10_1':
        return Lstm_32_32_10_1()
    if x == 'Lstm_32_32_20_1':
        return Lstm_32_32_20_1()
    if x == 'Lstm_32_64_1_1':
        return Lstm_32_64_1_1()
    if x == 'Lstm_32_64_2_1':
        return Lstm_32_64_2_1()
    if x == 'Lstm_32_64_5_1':
        return Lstm_32_64_5_1()
    if x == 'Lstm_32_64_10_1':
        return Lstm_32_64_10_1()
    if x == 'Lstm_32_64_20_1':
        return Lstm_32_64_20_1()
    if x == 'Lstm_32_128_1_1':
        return Lstm_32_128_1_1()
    if x == 'Lstm_32_128_2_1':
        return Lstm_32_128_2_1()
    if x == 'Lstm_32_128_5_1':
        return Lstm_32_128_5_1()
    if x == 'Lstm_32_128_10_1':
        return Lstm_32_128_10_1()
    if x == 'Lstm_32_128_20_1':
        return Lstm_32_128_20_1()
    if x == 'Lstm_32_256_1_1':
        return Lstm_32_256_1_1()
    if x == 'Lstm_32_256_2_1':
        return Lstm_32_256_2_1()
    if x == 'Lstm_32_256_5_1':
        return Lstm_32_256_5_1()
    if x == 'Lstm_32_256_10_1':
        return Lstm_32_256_10_1()
    if x == 'Lstm_32_256_20_1':
        return Lstm_32_256_20_1()
    if x == 'Lstm_64_16_1_1':
        return Lstm_64_16_1_1()
    if x == 'Lstm_64_16_2_1':
        return Lstm_64_16_2_1()
    if x == 'Lstm_64_16_5_1':
        return Lstm_64_16_5_1()
    if x == 'Lstm_64_16_10_1':
        return Lstm_64_16_10_1()
    if x == 'Lstm_64_16_20_1':
        return Lstm_64_16_20_1()
    if x == 'Lstm_64_32_1_1':
        return Lstm_64_32_1_1()
    if x == 'Lstm_64_32_2_1':
        return Lstm_64_32_2_1()
    if x == 'Lstm_64_32_5_1':
        return Lstm_64_32_5_1()
    if x == 'Lstm_64_32_10_1':
        return Lstm_64_32_10_1()
    if x == 'Lstm_64_32_20_1':
        return Lstm_64_32_20_1()
    if x == 'Lstm_64_64_1_1':
        return Lstm_64_64_1_1()
    if x == 'Lstm_64_64_2_1':
        return Lstm_64_64_2_1()
    if x == 'Lstm_64_64_5_1':
        return Lstm_64_64_5_1()
    if x == 'Lstm_64_64_10_1':
        return Lstm_64_64_10_1()
    if x == 'Lstm_64_64_20_1':
        return Lstm_64_64_20_1()
    if x == 'Lstm_64_128_1_1':
        return Lstm_64_128_1_1()
    if x == 'Lstm_64_128_2_1':
        return Lstm_64_128_2_1()
    if x == 'Lstm_64_128_5_1':
        return Lstm_64_128_5_1()
    if x == 'Lstm_64_128_10_1':
        return Lstm_64_128_10_1()
    if x == 'Lstm_64_128_20_1':
        return Lstm_64_128_20_1()
    if x == 'Lstm_64_256_1_1':
        return Lstm_64_256_1_1()
    if x == 'Lstm_64_256_2_1':
        return Lstm_64_256_2_1()
    if x == 'Lstm_64_256_5_1':
        return Lstm_64_256_5_1()
    if x == 'Lstm_64_256_10_1':
        return Lstm_64_256_10_1()
    if x == 'Lstm_64_256_20_1':
        return Lstm_64_256_20_1()
    if x == 'Lstm_128_16_1_1':
        return Lstm_128_16_1_1()
    if x == 'Lstm_128_16_2_1':
        return Lstm_128_16_2_1()
    if x == 'Lstm_128_16_5_1':
        return Lstm_128_16_5_1()
    if x == 'Lstm_128_16_10_1':
        return Lstm_128_16_10_1()
    if x == 'Lstm_128_16_20_1':
        return Lstm_128_16_20_1()
    if x == 'Lstm_128_32_1_1':
        return Lstm_128_32_1_1()
    if x == 'Lstm_128_32_2_1':
        return Lstm_128_32_2_1()
    if x == 'Lstm_128_32_5_1':
        return Lstm_128_32_5_1()
    if x == 'Lstm_128_32_10_1':
        return Lstm_128_32_10_1()
    if x == 'Lstm_128_32_20_1':
        return Lstm_128_32_20_1()
    if x == 'Lstm_128_64_1_1':
        return Lstm_128_64_1_1()
    if x == 'Lstm_128_64_2_1':
        return Lstm_128_64_2_1()
    if x == 'Lstm_128_64_5_1':
        return Lstm_128_64_5_1()
    if x == 'Lstm_128_64_10_1':
        return Lstm_128_64_10_1()
    if x == 'Lstm_128_64_20_1':
        return Lstm_128_64_20_1()
    if x == 'Lstm_128_128_1_1':
        return Lstm_128_128_1_1()
    if x == 'Lstm_128_128_2_1':
        return Lstm_128_128_2_1()
    if x == 'Lstm_128_128_5_1':
        return Lstm_128_128_5_1()
    if x == 'Lstm_128_128_10_1':
        return Lstm_128_128_10_1()
    if x == 'Lstm_128_128_20_1':
        return Lstm_128_128_20_1()
    if x == 'Lstm_128_256_1_1':
        return Lstm_128_256_1_1()
    if x == 'Lstm_128_256_2_1':
        return Lstm_128_256_2_1()
    if x == 'Lstm_128_256_5_1':
        return Lstm_128_256_5_1()
    if x == 'Lstm_128_256_10_1':
        return Lstm_128_256_10_1()
    if x == 'Lstm_128_256_20_1':
        return Lstm_128_256_20_1()
    if x == 'Lstm_256_16_1_1':
        return Lstm_256_16_1_1()
    if x == 'Lstm_256_16_2_1':
        return Lstm_256_16_2_1()
    if x == 'Lstm_256_16_5_1':
        return Lstm_256_16_5_1()
    if x == 'Lstm_256_16_10_1':
        return Lstm_256_16_10_1()
    if x == 'Lstm_256_16_20_1':
        return Lstm_256_16_20_1()
    if x == 'Lstm_256_32_1_1':
        return Lstm_256_32_1_1()
    if x == 'Lstm_256_32_2_1':
        return Lstm_256_32_2_1()
    if x == 'Lstm_256_32_5_1':
        return Lstm_256_32_5_1()
    if x == 'Lstm_256_32_10_1':
        return Lstm_256_32_10_1()
    if x == 'Lstm_256_32_20_1':
        return Lstm_256_32_20_1()
    if x == 'Lstm_256_64_1_1':
        return Lstm_256_64_1_1()
    if x == 'Lstm_256_64_2_1':
        return Lstm_256_64_2_1()
    if x == 'Lstm_256_64_5_1':
        return Lstm_256_64_5_1()
    if x == 'Lstm_256_64_10_1':
        return Lstm_256_64_10_1()
    if x == 'Lstm_256_64_20_1':
        return Lstm_256_64_20_1()
    if x == 'Lstm_256_128_1_1':
        return Lstm_256_128_1_1()
    if x == 'Lstm_256_128_2_1':
        return Lstm_256_128_2_1()
    if x == 'Lstm_256_128_5_1':
        return Lstm_256_128_5_1()
    if x == 'Lstm_256_128_10_1':
        return Lstm_256_128_10_1()
    if x == 'Lstm_256_128_20_1':
        return Lstm_256_128_20_1()
    if x == 'Lstm_256_256_1_1':
        return Lstm_256_256_1_1()
    if x == 'Lstm_256_256_2_1':
        return Lstm_256_256_2_1()
    if x == 'Lstm_256_256_5_1':
        return Lstm_256_256_5_1()
    if x == 'Lstm_256_256_10_1':
        return Lstm_256_256_10_1()
    if x == 'Lstm_256_256_20_1':
        return Lstm_256_256_20_1()
    if x == 'Lstm_16_16_1_2':
        return Lstm_16_16_1_2()
    if x == 'Lstm_16_16_2_2':
        return Lstm_16_16_2_2()
    if x == 'Lstm_16_16_5_2':
        return Lstm_16_16_5_2()
    if x == 'Lstm_16_16_10_2':
        return Lstm_16_16_10_2()
    if x == 'Lstm_16_16_20_2':
        return Lstm_16_16_20_2()
    if x == 'Lstm_16_32_1_2':
        return Lstm_16_32_1_2()
    if x == 'Lstm_16_32_2_2':
        return Lstm_16_32_2_2()
    if x == 'Lstm_16_32_5_2':
        return Lstm_16_32_5_2()
    if x == 'Lstm_16_32_10_2':
        return Lstm_16_32_10_2()
    if x == 'Lstm_16_32_20_2':
        return Lstm_16_32_20_2()
    if x == 'Lstm_16_64_1_2':
        return Lstm_16_64_1_2()
    if x == 'Lstm_16_64_2_2':
        return Lstm_16_64_2_2()
    if x == 'Lstm_16_64_5_2':
        return Lstm_16_64_5_2()
    if x == 'Lstm_16_64_10_2':
        return Lstm_16_64_10_2()
    if x == 'Lstm_16_64_20_2':
        return Lstm_16_64_20_2()
    if x == 'Lstm_16_128_1_2':
        return Lstm_16_128_1_2()
    if x == 'Lstm_16_128_2_2':
        return Lstm_16_128_2_2()
    if x == 'Lstm_16_128_5_2':
        return Lstm_16_128_5_2()
    if x == 'Lstm_16_128_10_2':
        return Lstm_16_128_10_2()
    if x == 'Lstm_16_128_20_2':
        return Lstm_16_128_20_2()
    if x == 'Lstm_16_256_1_2':
        return Lstm_16_256_1_2()
    if x == 'Lstm_16_256_2_2':
        return Lstm_16_256_2_2()
    if x == 'Lstm_16_256_5_2':
        return Lstm_16_256_5_2()
    if x == 'Lstm_16_256_10_2':
        return Lstm_16_256_10_2()
    if x == 'Lstm_16_256_20_2':
        return Lstm_16_256_20_2()
    if x == 'Lstm_32_16_1_2':
        return Lstm_32_16_1_2()
    if x == 'Lstm_32_16_2_2':
        return Lstm_32_16_2_2()
    if x == 'Lstm_32_16_5_2':
        return Lstm_32_16_5_2()
    if x == 'Lstm_32_16_10_2':
        return Lstm_32_16_10_2()
    if x == 'Lstm_32_16_20_2':
        return Lstm_32_16_20_2()
    if x == 'Lstm_32_32_1_2':
        return Lstm_32_32_1_2()
    if x == 'Lstm_32_32_2_2':
        return Lstm_32_32_2_2()
    if x == 'Lstm_32_32_5_2':
        return Lstm_32_32_5_2()
    if x == 'Lstm_32_32_10_2':
        return Lstm_32_32_10_2()
    if x == 'Lstm_32_32_20_2':
        return Lstm_32_32_20_2()
    if x == 'Lstm_32_64_1_2':
        return Lstm_32_64_1_2()
    if x == 'Lstm_32_64_2_2':
        return Lstm_32_64_2_2()
    if x == 'Lstm_32_64_5_2':
        return Lstm_32_64_5_2()
    if x == 'Lstm_32_64_10_2':
        return Lstm_32_64_10_2()
    if x == 'Lstm_32_64_20_2':
        return Lstm_32_64_20_2()
    if x == 'Lstm_32_128_1_2':
        return Lstm_32_128_1_2()
    if x == 'Lstm_32_128_2_2':
        return Lstm_32_128_2_2()
    if x == 'Lstm_32_128_5_2':
        return Lstm_32_128_5_2()
    if x == 'Lstm_32_128_10_2':
        return Lstm_32_128_10_2()
    if x == 'Lstm_32_128_20_2':
        return Lstm_32_128_20_2()
    if x == 'Lstm_32_256_1_2':
        return Lstm_32_256_1_2()
    if x == 'Lstm_32_256_2_2':
        return Lstm_32_256_2_2()
    if x == 'Lstm_32_256_5_2':
        return Lstm_32_256_5_2()
    if x == 'Lstm_32_256_10_2':
        return Lstm_32_256_10_2()
    if x == 'Lstm_32_256_20_2':
        return Lstm_32_256_20_2()
    if x == 'Lstm_64_16_1_2':
        return Lstm_64_16_1_2()
    if x == 'Lstm_64_16_2_2':
        return Lstm_64_16_2_2()
    if x == 'Lstm_64_16_5_2':
        return Lstm_64_16_5_2()
    if x == 'Lstm_64_16_10_2':
        return Lstm_64_16_10_2()
    if x == 'Lstm_64_16_20_2':
        return Lstm_64_16_20_2()
    if x == 'Lstm_64_32_1_2':
        return Lstm_64_32_1_2()
    if x == 'Lstm_64_32_2_2':
        return Lstm_64_32_2_2()
    if x == 'Lstm_64_32_5_2':
        return Lstm_64_32_5_2()
    if x == 'Lstm_64_32_10_2':
        return Lstm_64_32_10_2()
    if x == 'Lstm_64_32_20_2':
        return Lstm_64_32_20_2()
    if x == 'Lstm_64_64_1_2':
        return Lstm_64_64_1_2()
    if x == 'Lstm_64_64_2_2':
        return Lstm_64_64_2_2()
    if x == 'Lstm_64_64_5_2':
        return Lstm_64_64_5_2()
    if x == 'Lstm_64_64_10_2':
        return Lstm_64_64_10_2()
    if x == 'Lstm_64_64_20_2':
        return Lstm_64_64_20_2()
    if x == 'Lstm_64_128_1_2':
        return Lstm_64_128_1_2()
    if x == 'Lstm_64_128_2_2':
        return Lstm_64_128_2_2()
    if x == 'Lstm_64_128_5_2':
        return Lstm_64_128_5_2()
    if x == 'Lstm_64_128_10_2':
        return Lstm_64_128_10_2()
    if x == 'Lstm_64_128_20_2':
        return Lstm_64_128_20_2()
    if x == 'Lstm_64_256_1_2':
        return Lstm_64_256_1_2()
    if x == 'Lstm_64_256_2_2':
        return Lstm_64_256_2_2()
    if x == 'Lstm_64_256_5_2':
        return Lstm_64_256_5_2()
    if x == 'Lstm_64_256_10_2':
        return Lstm_64_256_10_2()
    if x == 'Lstm_64_256_20_2':
        return Lstm_64_256_20_2()
    if x == 'Lstm_128_16_1_2':
        return Lstm_128_16_1_2()
    if x == 'Lstm_128_16_2_2':
        return Lstm_128_16_2_2()
    if x == 'Lstm_128_16_5_2':
        return Lstm_128_16_5_2()
    if x == 'Lstm_128_16_10_2':
        return Lstm_128_16_10_2()
    if x == 'Lstm_128_16_20_2':
        return Lstm_128_16_20_2()
    if x == 'Lstm_128_32_1_2':
        return Lstm_128_32_1_2()
    if x == 'Lstm_128_32_2_2':
        return Lstm_128_32_2_2()
    if x == 'Lstm_128_32_5_2':
        return Lstm_128_32_5_2()
    if x == 'Lstm_128_32_10_2':
        return Lstm_128_32_10_2()
    if x == 'Lstm_128_32_20_2':
        return Lstm_128_32_20_2()
    if x == 'Lstm_128_64_1_2':
        return Lstm_128_64_1_2()
    if x == 'Lstm_128_64_2_2':
        return Lstm_128_64_2_2()
    if x == 'Lstm_128_64_5_2':
        return Lstm_128_64_5_2()
    if x == 'Lstm_128_64_10_2':
        return Lstm_128_64_10_2()
    if x == 'Lstm_128_64_20_2':
        return Lstm_128_64_20_2()
    if x == 'Lstm_128_128_1_2':
        return Lstm_128_128_1_2()
    if x == 'Lstm_128_128_2_2':
        return Lstm_128_128_2_2()
    if x == 'Lstm_128_128_5_2':
        return Lstm_128_128_5_2()
    if x == 'Lstm_128_128_10_2':
        return Lstm_128_128_10_2()
    if x == 'Lstm_128_128_20_2':
        return Lstm_128_128_20_2()
    if x == 'Lstm_128_256_1_2':
        return Lstm_128_256_1_2()
    if x == 'Lstm_128_256_2_2':
        return Lstm_128_256_2_2()
    if x == 'Lstm_128_256_5_2':
        return Lstm_128_256_5_2()
    if x == 'Lstm_128_256_10_2':
        return Lstm_128_256_10_2()
    if x == 'Lstm_128_256_20_2':
        return Lstm_128_256_20_2()
    if x == 'Lstm_256_16_1_2':
        return Lstm_256_16_1_2()
    if x == 'Lstm_256_16_2_2':
        return Lstm_256_16_2_2()
    if x == 'Lstm_256_16_5_2':
        return Lstm_256_16_5_2()
    if x == 'Lstm_256_16_10_2':
        return Lstm_256_16_10_2()
    if x == 'Lstm_256_16_20_2':
        return Lstm_256_16_20_2()
    if x == 'Lstm_256_32_1_2':
        return Lstm_256_32_1_2()
    if x == 'Lstm_256_32_2_2':
        return Lstm_256_32_2_2()
    if x == 'Lstm_256_32_5_2':
        return Lstm_256_32_5_2()
    if x == 'Lstm_256_32_10_2':
        return Lstm_256_32_10_2()
    if x == 'Lstm_256_32_20_2':
        return Lstm_256_32_20_2()
    if x == 'Lstm_256_64_1_2':
        return Lstm_256_64_1_2()
    if x == 'Lstm_256_64_2_2':
        return Lstm_256_64_2_2()
    if x == 'Lstm_256_64_5_2':
        return Lstm_256_64_5_2()
    if x == 'Lstm_256_64_10_2':
        return Lstm_256_64_10_2()
    if x == 'Lstm_256_64_20_2':
        return Lstm_256_64_20_2()
    if x == 'Lstm_256_128_1_2':
        return Lstm_256_128_1_2()
    if x == 'Lstm_256_128_2_2':
        return Lstm_256_128_2_2()
    if x == 'Lstm_256_128_5_2':
        return Lstm_256_128_5_2()
    if x == 'Lstm_256_128_10_2':
        return Lstm_256_128_10_2()
    if x == 'Lstm_256_128_20_2':
        return Lstm_256_128_20_2()
    if x == 'Lstm_256_256_1_2':
        return Lstm_256_256_1_2()
    if x == 'Lstm_256_256_2_2':
        return Lstm_256_256_2_2()
    if x == 'Lstm_256_256_5_2':
        return Lstm_256_256_5_2()
    if x == 'Lstm_256_256_10_2':
        return Lstm_256_256_10_2()
    if x == 'Lstm_256_256_20_2':
        return Lstm_256_256_20_2()


if __name__ == "__main__":
    main()

import time
import argparse

import tqdm
import torch
from torch.utils.data import DataLoader
from humanfriendly import format_timespan

from model import *
from dataset import HDF5Dataset


def main():
    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Default_model', help="Name of model to eval")
    parser.add_argument("--classifier", default='C', help="Choose classifier architecture, C, CBN")
    parser.add_argument("--train_path", default='Train_data.hdf5', help="HDF5 train Dataset path")
    parser.add_argument("--test_path", default='Test_data.hdf5', help="HDF5 test Dataset path")
    parser.add_argument("--batch_size", type=int, default=4, help="Size of the batches")
    parser.add_argument("--thresh", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

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

    # Load parameters from trained model
    net.load_state_dict(torch.load('../models/' + args.model_name + '.pth'))
    net.eval()

    # Evaluate model on training dataset

    # True/False Positives/Negatives
    correct = 0
    total = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    with tqdm.tqdm(total=len(train_loader), desc='Train dataset evaluation', position=0) as train_bar:
        with torch.no_grad():
            for data in train_loader:
                traces, labels = data[0].to(device), data[1].to(device)
                outputs = net(traces)
                predicted = (outputs > args.thresh)
                total += labels.size(0)

                for i, pred in enumerate(predicted):
                    if pred:
                        if pred == labels[i]:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if pred == labels[i]:
                            tn += 1
                        else:
                            fn += 1

                correct += (predicted == labels).sum().item()
                train_bar.update(1)

    # Evaluation metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)

    eval_1 = time.time()
    ev_1 = eval_1 - start_time

    print(f'Evaluation details: \n\n{args}\n'
          f'Training Evaluation results: \n\n\n'
          f'correct: {correct}, total: {total}\n\n'
          f'True positives: {tp}\n'
          f'False positives: {fp}\n'
          f'True negatives: {tn}\n'
          f'False negatives: {fn}\n\n'
          f'Evaluation metrics:\n\n'          
          f'Precision: {precision:5.3f}\n'
          f'Recall: {recall:5.3f}\n'
          f'F-score: {f_score:5.3f}\n')

    print('Accuracy of the network on the train set: %d %%\n' % (100 * correct / total))

    # Evaluate model on test set

    # True/False Positives/Negatives
    correct = 0
    total = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    with tqdm.tqdm(total=len(test_loader), desc='Test dataset evaluation', position=0) as test_bar:
        with torch.no_grad():
            for data in test_loader:
                traces, labels = data[0].to(device), data[1].to(device)
                outputs = net(traces)
                predicted = (outputs > args.thresh)
                total += labels.size(0)

                for i, pred in enumerate(predicted):
                    if pred:
                        if pred == labels[i]:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if pred == labels[i]:
                            tn += 1
                        else:
                            fn += 1

                correct += (predicted == labels).sum().item()
                test_bar.update(1)

    # Evaluation metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)

    eval_2 = time.time()
    ev_2 = eval_2 - eval_1
    ev_t = eval_2 - start_time

    print(f'Test Evaluation results: \n\n\n'
          f'correct: {correct}, total: {total}\n\n'
          f'True positives: {tp}\n'
          f'False positives: {fp}\n'
          f'True negatives: {tn}\n'
          f'False negatives: {fn}\n\n'
          f'Evaluation metrics:\n\n'
          f'Precision: {precision:5.3f}\n'
          f'Recall: {recall:5.3f}\n'
          f'F-score: {f_score:5.3f}\n\n'
          f'Training evaluation time: {format_timespan(ev_1)}\n'
          f'Test evaluation time: {format_timespan(ev_2)}\n'
          f'Total execution time: {format_timespan(ev_t)}\n\n')

    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
    print(f'Number of network parameters: {nparams}')


def get_classifier(x):
    return {
        'C': ClassConv(),
        'CBN': ClassConvBN(),
        'CBN_v2': CBN_v2(),
    }.get(x, ClassConv())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()

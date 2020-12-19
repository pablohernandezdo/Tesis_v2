import time
import argparse
import itertools
from pathlib import Path

import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from humanfriendly import format_timespan


from model import *
from dataset import HDF5Dataset


def main():
    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='defaultmodel', help="Name of model to eval")
    parser.add_argument("--model_folder", default='default', help="Folder to save model")
    parser.add_argument("--classifier", default='1h6k', help="Choose classifier architecture")
    parser.add_argument("--test_path", default='Test_data.hdf5', help="HDF5 test Dataset path")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--beta", type=float, default=2, help="Fscore beta parameter")
    args = parser.parse_args()

    # Create curves folders
    Path(f"../Analysis/Confusion_matrices/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/PR_curves/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/ROC_curves/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/Fscore_curves/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/FPFN_curves/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/Histograms/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/Output_values/{args.model_folder}").mkdir(parents=True, exist_ok=True)

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # Print number of network parameters
    print(f'Number of network parameters: {nparams}\n')

    # Seismic and non seismic output values
    hist = 1
    s_outputs = []
    ns_outputs = []

    # Preallocate precision and recall values
    precision = []
    fp_rate = []
    recall = []
    fscores = []

    fp_plt = []
    fn_plt = []
    # Confusion matrix
    cm = []

    # Record max fscore value obtained
    max_fscore = 0

    # Record threshold of best fscore
    best_thresh = 0

    # Thresholds to evaluate performance on
    thresholds = np.arange(0.025, 1, 0.025)
    # thresholds = [0, 0.5, 0.9]

    # Round threshold values
    thresholds = np.around(thresholds, decimals=3)

    # Evaluate model on training dataset

    for thresh in thresholds:

        # True/False Positives/Negatives
        correct = 0
        total = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        # Print threshold value
        print(f'Threshold value: {thresh}\n')

        # Evaluate
        with tqdm.tqdm(total=len(test_loader), desc='Test dataset evaluation') as test_bar:
            with torch.no_grad():
                for data in test_loader:
                    traces, labels = data[0].to(device), data[1].to(device)
                    outputs = net(traces)
                    predicted = (outputs > thresh)
                    total += labels.size(0)

                    # Add output values to list (just once)
                    if hist:
                        for i, lab in enumerate(labels):
                            if lab:
                                s_outputs.append(outputs[i].item())

                            else:
                                ns_outputs.append(outputs[i].item())

                    # Count true positives, true negatives, etc.
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
                    test_bar.update()

        # Run just one time
        hist = 0

        # Metrics
        pre, rec, fpr, fscore = print_metrics(tp, fp, tn, fn, args.beta)
        recall.append(rec)
        fp_rate.append(fpr)
        precision.append(pre)
        fscores.append(fscore)

        fp_plt.append(fp)
        fn_plt.append(fn)

        # Save best conf matrix
        if fscore > max_fscore:
            max_fscore = fscore
            cm = np.asarray([[tp, fn], [fp, tn]])
            best_thresh = thresh

        eval_1 = time.time()
        ev_1 = eval_1 - start_time

        print(f'Test evaluation time: {format_timespan(ev_1)}\n')

    # Add point (0, 1) to PR curve
    precision.append(1)
    recall.append(0)

    # Add point (1, 0.5) to PR curve
    precision.insert(0, 0.5)
    recall.insert(0, 1)

    # Add point (0, 0)  to ROC curve
    fp_rate.append(0)

    # Add point (1, 1) to ROC curve
    fp_rate.insert(0, 1)

    # Area under curve
    pr_auc = np.trapz(precision[::-1], x=recall[::-1])
    roc_auc = np.trapz(recall[::-1], x=fp_rate[::-1])

    # Print fscores
    print(f'Best test threshold: {best_thresh}, f-score: {max_fscore:5.3f}\n\n'
          f'Test PR AUC: {pr_auc:5.3f}\n'
          f'Test ROC AUC: {roc_auc:5.3f}')

    # Save output values to file
    with open(f'../Analysis/Output_values/{args.model_folder}/outputs_{args.model_name}.txt', 'w') as f:
        f.write('Seismic outputs\n')
        f.write('\n'.join(list(map(str, s_outputs))))
        f.write('\nNon-Seismic outputs\n')
        f.write('\n'.join(list(map(str, ns_outputs))))

    # Plot histograms
    plot_histograms(s_outputs, ns_outputs, args.model_folder, args.model_name)

    # Plot best confusion matrices
    target_names = ['Seismic', 'Non Seismic']

    # Confusion matrix
    plot_confusion_matrix(cm, target_names,
                          title=f'Confusion matrix {args.model_name}, threshold = {best_thresh}',
                          filename=f'../Analysis/Confusion_matrices/{args.model_folder}/Confusion_matrix_STEAD_{args.model_name}.png')

    # F-score vs thresholds curve
    plt.figure()
    plt.plot(thresholds, fscores)
    plt.title(f'Fscores por umbral modelo {args.model_name}')
    plt.xlabel('Umbrales')
    plt.ylabel('F-score')
    plt.grid(True)
    plt.savefig(f'../Analysis/Fscore_curves/{args.model_folder}/Fscore_{args.model_name}.png')

    # False positives / False negatives curve
    plt.figure()
    line_fp, = plt.plot(thresholds, fp_plt, label='False positives')
    line_fn, = plt.plot(thresholds, fn_plt, label='False negatives')

    plt.title(f'FP y FN modelo {args.model_name}')
    plt.xlabel('Umbrales')
    plt.ylabel('Total')
    plt.grid(True)
    plt.legend(handles=[line_fp, line_fn], loc='best')
    plt.savefig(f'../Analysis/FPFN_curves/{args.model_folder}/FPFN_{args.model_name}.png')

    # Precision/Recall curve test dataset
    plt.figure()
    plt.plot(recall, precision)

    # Annotate threshold values
    for i, j, k in zip(recall, precision, thresholds):
        plt.annotate(str(k), (i, j))

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'PR test dataset curve for model {args.model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.02, 1.02)
    plt.ylim(0.48, 1.02)
    plt.grid(True)
    plt.savefig(f'../Analysis/PR_curves/{args.model_folder}/PR_test_{args.model_name}.png')

    # Receiver operating characteristic curve test dataset
    plt.figure()
    plt.plot(fp_rate, recall)

    # Annotate
    for i, j, k in zip(fp_rate, recall, thresholds):
        plt.annotate(str(k), (i, j))

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC test dataset curve for model {args.model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.savefig(f'../Analysis/ROC_curves/{args.model_folder}/ROC_test_{args.model_name}.png')


def plot_histograms(s_outputs, ns_outputs, model_folder, model_name):

    plt.figure()

    n_seis, bins_seis, patches_seis = plt.hist(s_outputs, bins=100, color='blue', alpha=0.6, label='Seismic')
    n_nseis, bins_nseis, patches_nseis = plt.hist(ns_outputs, bins=100, color='red', alpha=0.6, label='Non seismic')

    plt.title(f'Output values histogram model {model_name}')
    plt.xlabel('Net output value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(f'../Analysis/Histograms/{model_folder}/Histogram_{model_name}.png')


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                          filename='Confusion_matrix.png', cmap=None, normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    missclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, missclass))
    plt.savefig(filename)


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

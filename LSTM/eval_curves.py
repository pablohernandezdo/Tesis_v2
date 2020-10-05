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
    # Create curves folders
    Path("../Confusion_matrices").mkdir(exist_ok=True)
    Path("../PR_curves").mkdir(exist_ok=True)
    Path("../ROC_curves").mkdir(exist_ok=True)

    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='XXL_lr0000001_bs32', help="Name of model to eval")
    parser.add_argument("--classifier", default='XXL', help="Choose classifier architecture, C, S, XS, XL, XXL, XXXL")
    parser.add_argument("--train_path", default='Train_data.hdf5', help="HDF5 train Dataset path")
    parser.add_argument("--test_path", default='Test_data.hdf5', help="HDF5 test Dataset path")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
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

    # Load from trained model
    net.load_state_dict(torch.load('../models/' + args.model_name + '.pth'))
    net.eval()

    # Print number of network parameters
    print(f'Number of network parameters: {nparams}\n')

    # Preallocate precision and recall values
    tr_precision = []
    tst_precision = []
    tr_fp_rate = []
    tst_fp_rate = []
    tr_recall = []
    tst_recall = []
    tr_fscores = []
    tst_fscores = []

    # Confusion matrix
    tr_cm = []
    tst_cm = []

    # Record max fscore value obtained
    tr_max_fscore = 0
    tst_max_fscore = 0

    # Record threshold of best fscore
    tr_best_thresh = 0
    tst_best_thresh = 0

    # Thresholds to evaluate performance on
    thresholds = np.arange(0.05, 1, 0.05)

    # Round threshold values
    thresholds = np.around(thresholds, decimals=2)

    # Evaluate model on training dataset

    for thresh in thresholds:

        # True/False Positives/Negatives
        correct = 0
        total = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        # Print threshold value
        print(f'Threshold value: {thresh}\n')

        # Evaluate
        with tqdm.tqdm(total=len(train_loader), desc='Train dataset evaluation', position=0) as train_bar:
            with torch.no_grad():
                for data in train_loader:
                    traces, labels = data[0].to(device), data[1].to(device)
                    outputs = net(traces)
                    predicted = (outputs > thresh)
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

        # Metrics
        tr_pre, tr_rec, tr_fpr, tr_fscore = print_metrics(tp, fp, tn, fn)
        tr_recall.append(tr_rec)
        tr_fp_rate.append(tr_fpr)
        tr_precision.append(tr_pre)
        tr_fscores.append(tr_fscore)

        # Save best conf matrix
        if tr_fscore > tr_max_fscore:
            tr_max_fscore = tr_fscore
            tr_cm = np.asarray([[tp, fn], [fp, tn]])
            tr_best_thresh = thresh

        eval_1 = time.time()
        ev_1 = eval_1 - start_time

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
                    predicted = (outputs > thresh)
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

        # Metrics
        tst_pre, tst_rec, tst_fpr, tst_fscore = print_metrics(tp, fp, tn, fn)
        tst_recall.append(tst_rec)
        tst_fp_rate.append(tst_fpr)
        tst_precision.append(tst_pre)
        tst_fscores.append(tst_fscore)

        # Save best conf matrix
        if tst_fscore > tst_max_fscore:
            tst_max_fscore = tst_fscore
            tst_cm = np.asarray([[tp, fn], [fp, tn]])
            tst_best_thresh = thresh

        eval_2 = time.time()
        ev_2 = eval_2 - eval_1
        ev_t = eval_2 - start_time

        print(f'Training evaluation time: {format_timespan(ev_1)}\n'
              f'Test evaluation time: {format_timespan(ev_2)}\n'
              f'Total execution time: {format_timespan(ev_t)}\n\n')

    # Area under curve
    tr_pr_auc = np.trapz(tr_precision, x=tr_recall[::-1])
    tst_pr_auc = np.trapz(tst_precision, x=tst_recall[::-1])

    tr_roc_auc = np.trapz(tr_recall, x=tr_fp_rate[::-1])
    tst_roc_auc = np.trapz(tst_recall, x=tst_fp_rate[::-1])

    # Print fscores
    print(f'Best train threshold: {tr_best_thresh}, f-score: {tr_max_fscore:5.3f}\n'
          f'Best test threshold: {tst_best_thresh}, f-score: {tst_max_fscore:5.3f}\n\n'
          f'Train PR AUC: {tr_pr_auc:5.3f}\n'
          f'Test PR AUC: {tst_pr_auc:5.3f}\n\n'
          f'Train ROC AUC: {tr_roc_auc:5.3f}\n'
          f'Test ROC AUC: {tst_roc_auc:5.3f}\n')

    # Plot best confusion matrices
    target_names = ['Seismic', 'Non Seismic']

    # Confusion matrix
    plot_confusion_matrix(tr_cm, target_names,
                          title=f'Confusion matrix {args.model_name} train, threshold = {tr_best_thresh}',
                          filename=f'../Confusion_matrices/Confusion_matrix_train_{args.model_name}.png')

    # Confusion matrix
    plot_confusion_matrix(tst_cm, target_names,
                          title=f'Confusion matrix {args.model_name} test, threshold = {tst_best_thresh}',
                          filename=f'../Confusion_matrices/Confusion_matrix_test_{args.model_name}.png')

    # Precision/Recall curve train dataset
    plt.figure()
    plt.plot(tr_recall, tr_precision)

    # Annotate threshold values
    for i, j, k in zip(tr_recall, tr_precision, thresholds):
        plt.annotate(str(k), (i, j))

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'PR train dataset curve for model {args.model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'../PR_curves/PR_train_{args.model_name}.png')

    # Receiver operating characteristic curve train dataset
    plt.figure()
    plt.plot(tr_fp_rate, tr_recall)

    # Annotate
    for i, j, k in zip(tr_fp_rate, tr_recall, thresholds):
        plt.annotate(str(k), (i, j))

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC train dataset curve for model {args.model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'../ROC_curves/ROC_train_{args.model_name}.png')

    # Precision/Recall curve test dataset
    plt.figure()
    plt.plot(tst_recall, tst_precision)

    # Annotate threshold values
    for i, j, k in zip(tst_recall, tst_precision, thresholds):
        plt.annotate(str(k), (i, j))

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'PR test dataset curve for model {args.model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'../PR_curves/PR_test_{args.model_name}.png')

    # Receiver operating characteristic curve test dataset
    plt.figure()
    plt.plot(tst_fp_rate, tst_recall)

    # Annotate
    for i, j, k in zip(tst_fp_rate, tst_recall, thresholds):
        plt.annotate(str(k), (i, j))

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC test dataset curve for model {args.model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'../ROC_curves/ROC_test_{args.model_name}.png')


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                          filename='Confusion_matrix.png', cmap=None, normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

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
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(filename)


def print_metrics(tp, fp, tn, fn):

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
        fscore = 2 * (precision * recall) / (precision + recall)

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


def get_classifier(x):
    return {
        'LSTM': CNNLSTMANN(),
        'LSTM_v2': CNNLSTMANN_v2(),
    }.get(x, CNNLSTMANN())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()

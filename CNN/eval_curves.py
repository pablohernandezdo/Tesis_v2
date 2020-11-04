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
    Path("../FPFN_curves").mkdir(exist_ok=True)

    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='defaultmodel', help="Name of model to eval")
    parser.add_argument("--classifier", default='1h6k', help="Choose classifier architecture")
    parser.add_argument("--test_path", default='Test_data.hdf5', help="HDF5 test Dataset path")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    args = parser.parse_args()

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
    net.load_state_dict(torch.load('../models/' + args.model_name + '.pth'))
    net.eval()

    # Print number of network parameters
    print(f'Number of network parameters: {nparams}\n')

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
        with tqdm.tqdm(total=len(test_loader), desc='Test dataset evaluation') as test_bar:
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
                    test_bar.update()

        # Metrics
        pre, rec, fpr, fscore = print_metrics(tp, fp, tn, fn)
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
    pr_auc = np.trapz(precision, x=recall[::-1])
    roc_auc = np.trapz(recall, x=fp_rate[::-1])

    # Print fscores
    print(f'Best train threshold: {best_thresh}, f-score: {max_fscore:5.3f}\n'
          f'Best test threshold: {best_thresh}, f-score: {max_fscore:5.3f}\n\n'
          f'Test PR AUC: {pr_auc:5.3f}\n'
          f'Test ROC AUC: {roc_auc:5.3f}')

    # Plot best confusion matrices
    target_names = ['Seismic', 'Non Seismic']

    # Confusion matrix
    plot_confusion_matrix(cm, target_names,
                          title=f'Confusion matrix {args.model_name} train, threshold = {best_thresh}',
                          filename=f'../Confusion_matrices/Confusion_matrix_test_{args.model_name}.png')

    # Fale positives / False negatives curve
    plt.figure()
    line_fp, = plt.plot(thresholds, fp_plt, label='False positives')
    line_fn, = plt.plot(thresholds, fn_plt, label='False negatives')

    plt.title(f'FP y FN modelo {args.model_name}')
    plt.xlabel('Umbrales')
    plt.ylabel('Total')
    plt.grid(True)
    plt.legend(handles=[line_fp, line_fn], loc='best')
    plt.savefig(f'../FPFN_curves/FPFN_{args.model_name}.png')

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
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'../PR_curves/PR_test_{args.model_name}.png')

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
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'../ROC_curves/ROC_test_{args.model_name}.png')


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
    if x == '1c1h':
        return CNN1P1H1h()
    if x == '1c2h':
        return CNN1P1H2h()
    if x == '1c5h':
        return CNN1P1H5h()
    if x == '1c1k':
        return CNN1P1H1k()
    if x == '1c2k':
        return CNN1P1H2k()
    if x == '1c3k':
        return CNN1P1H3k()
    if x == '1c4k':
        return CNN1P1H4k()
    if x == '1c5k':
        return CNN1P1H5k()
    if x == '1c10k':
        return CNN1P1H10k()
    if x == '1c10k10k':
        return CNN1P2H10k10k()
    if x == '1c10k5k':
        return CNN1P2H10k5k()
    if x == '1c10k1k':
        return CNN1P2H10k1k()
    if x == '1c10k1h':
        return CNN1P2H10k1h()
    if x == '1c10k10':
        return CNN1P2H10k10()
    if x == '1c6k6k':
        return CNN1P2H6k6k()
    if x == '1c6k1k':
        return CNN1P2H6k1k()
    if x == '1c6k1h':
        return CNN1P2H6k1h()
    if x == '1c6k10':
        return CNN1P2H6k10()
    if x == '1c5k5k':
        return CNN1P2H5k5k()
    if x == '1c5k1k':
        return CNN1P2H5k1k()
    if x == '1c5k1h':
        return CNN1P2H5k1h()
    if x == '1c5k10':
        return CNN1P2H5k10()
    if x == '1c4k4k':
        return CNN1P2H4k4k()
    if x == '1c4k1k':
        return CNN1P2H4k1k()
    if x == '1c4k1h':
        return CNN1P2H4k1h()
    if x == '1c4k10':
        return CNN1P2H4k10()
    if x == '1c3k3k':
        return CNN1P2H3k3k()
    if x == '1c3k1k':
        return CNN1P2H3k1k()
    if x == '1c3k1h':
        return CNN1P2H3k1h()
    if x == '1c3k10':
        return CNN1P2H3k10()
    if x == '1c2k2k':
        return CNN1P2H2k2k()
    if x == '1c2k1k':
        return CNN1P2H2k1k()
    if x == '1c2k1h':
        return CNN1P2H2k1h()
    if x == '1c2k10':
        return CNN1P2H2k10()
    if x == '1c1k1k':
        return CNN1P2H1k1k()
    if x == '1c1k1h':
        return CNN1P2H1k1h()
    if x == '1c1k10':
        return CNN1P2H1k10()
    if x == '1c5h5h':
        return CNN1P2H5h5h()
    if x == '1c5h1h':
        return CNN1P2H5h1h()
    if x == '1c5h10':
        return CNN1P2H5h10()
    if x == '1c2h2h':
        return CNN1P2H2h2h()
    if x == '1c2h1h':
        return CNN1P2H2h1h()
    if x == '1c2h10':
        return CNN1P2H2h10()
    if x == '1c1h1h':
        return CNN1P2H1h1h()
    if x == '1c1h10':
        return CNN1P2H1h10()
    if x == '2c20k':
        return CNN2P1H20k()
    if x == '2c15k':
        return CNN2P1H15k()
    if x == '2c10k':
        return CNN2P1H10k()
    if x == '2c5k':
        return CNN2P1H5k()
    if x == '2c3k':
        return CNN2P1H3k()
    if x == '2c2k':
        return CNN2P1H2k()
    if x == '2c1k':
        return CNN2P1H1k()
    if x == '2c20k20k':
        return CNN2P1H20k20k()
    if x == '2c20k10k':
        return CNN2P1H20k10k()
    if x == '2c20k5k':
        return CNN2P1H20k5k()
    if x == '2c20k2k':
        return CNN2P1H20k2k()
    if x == '2c20k1k':
        return CNN2P1H20k1k()
    if x == '2c20k5h':
        return CNN2P1H20k5h()
    if x == '2c20k1h':
        return CNN2P1H20k1h()
    if x == '2c20k10':
        return CNN2P1H20k10()
    if x == '2c15k15k':
        return CNN2P1H15k15k()
    if x == '2c15k10k':
        return CNN2P1H15k10k()
    if x == '2c15k5k':
        return CNN2P1H15k5k()
    if x == '2c15k2k':
        return CNN2P1H15k2k()
    if x == '2c15k1k':
        return CNN2P1H15k1k()
    if x == '2c15k1h':
        return CNN2P1H15k1h()
    if x == '2c15k10':
        return CNN2P1H15k10()
    if x == '2c10k10k':
        return CNN2P1H10k10k()
    if x == '2c10k5k':
        return CNN2P1H10k5k()
    if x == '2c10k2k':
        return CNN2P1H10k2k()
    if x == '2c10k1k':
        return CNN2P1H10k1k()
    if x == '2c10k5h':
        return CNN2P1H10k5h()
    if x == '2c10k1h':
        return CNN2P1H10k1h()
    if x == '2c10k10':
        return CNN2P1H10k10()
    if x == '2c5k5k':
        return CNN2P1H5k5k()
    if x == '2c5k2k':
        return CNN2P1H5k2k()
    if x == '2c5k1k':
        return CNN2P1H5k1k()
    if x == '2c5k5h':
        return CNN2P1H5k5h()
    if x == '2c5k1h':
        return CNN2P1H5k1h()
    if x == '2c5k10':
        return CNN2P1H5k10()
    if x == '2c3k3k':
        return CNN2P1H3k3k()
    if x == '2c3k2k':
        return CNN2P1H3k2k()
    if x == '2c3k1k':
        return CNN2P1H3k1k()
    if x == '2c3k5h':
        return CNN2P1H3k5h()
    if x == '2c3k1h':
        return CNN2P1H3k1h()
    if x == '2c3k10':
        return CNN2P1H3k10()
    if x == '2c2k2k':
        return CNN2P1H2k2k()
    if x == '2c2k1k':
        return CNN2P1H2k1k()
    if x == '2c2k5h':
        return CNN2P1H2k5h()
    if x == '2c2k1h':
        return CNN2P1H2k1h()
    if x == '2c2k10':
        return CNN2P1H2k10()
    if x == '2c1k1k':
        return CNN2P1H1k1k()
    if x == '2c1k5h':
        return CNN2P1H1k5h()
    if x == '2c1k1h':
        return CNN2P1H1k1h()
    if x == '2c1k10':
        return CNN2P1H1k10()
    else:
        return CNN2P1H1k10()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()

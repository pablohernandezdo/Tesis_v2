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

    # Output values
    output_values = []

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
    # thresholds = np.arange(0.05, 1, 0.05)
    thresholds = [0.5]

    # Round threshold values
    thresholds = np.around(thresholds, decimals=2)

    # Evaluate model on training dataset

    for thresh in thresholds:

        # True/False Positives/Negatives
        correct = 0
        total = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        # Intermediate output values lists

        s_outputs = []
        ns_outputs = []

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

                    # Add output values to list
                    for i, lab in enumerate(labels):
                        if lab:
                            s_outputs.append(outputs[i].cpu().numpy())

                        else:
                            ns_outputs.append(outputs[i].cpu().numpy())

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

        # Append outputs to general list
        output_values.append([s_outputs, ns_outputs, thresh])

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

    # Plot histograms
    plot_histograms(output_values, args.model_folder, args.model_name)

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


def plot_histograms(output_values, model_folder, model_name):

    plt.figure()

    for seismic_outputs, nseismic_outputs, thresh in output_values:

        plt.clf()

        n_seis, bins_seis, patches_seis = plt.hist(seismic_outputs, 10, facecolor='blue')
        # n_nseis, bins_nseis, patches_nseis = plt.hist(nseismic_outputs, bins='auto', facecolor='red')

        plt.xlabel('rango ?')
        plt.ylabel('Probabilitiiii')
        plt.grid(True)

        plt.savefig(f'../Analysis/Histograms/{model_folder}/Histogram_{model_name}_{thresh}.png')


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

import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder', default='Analysis/CSVOutputs/Test/step4',
                        help='Path to CSV files folder')
    parser.add_argument('--xls_name',
                        help='Best models excel file name')
    parser.add_argument("--beta", type=float, default=2,
                        help="Fscore beta parameter")
    args = parser.parse_args()

    # Create figures and excel files folder
    Path("Analysis/Excel_reports").mkdir(parents=True, exist_ok=True)

    Path(f"Analysis/Fscore").mkdir(parents=True, exist_ok=True)

    Path(f"Analysis/Histogram").mkdir(parents=True, exist_ok=True)

    Path(f"Analysis/FPFN/").mkdir(parents=True, exist_ok=True)

    Path(f"Analysis/PR/").mkdir(parents=True, exist_ok=True)

    Path(f"Analysis/ROC/").mkdir(parents=True, exist_ok=True)

    # Define threshold to evaluate on
    thresholds = np.arange(0, 1, 0.01)

    # Model metrics list

    model_names = []
    prec_list = []
    rec_list = []
    fpr_list = []
    fscore_list = []

    fp_list = []
    fn_list = []

    # Read Csv files
    for csv in os.listdir(args.csv_folder):

        if os.path.splitext(csv)[-1] != '.csv':
            continue

        # Read csv file
        df = pd.read_csv(f'{args.csv_folder}/{csv}')

        # Get model name
        model_name = os.path.splitext(csv)[0]

        # Preallocate variables
        prec = np.zeros(len(thresholds))
        rec = np.zeros(len(thresholds))
        fpr = np.zeros(len(thresholds))
        fscore = np.zeros(len(thresholds))

        fp_arr = np.zeros(len(thresholds))
        fn_arr = np.zeros(len(thresholds))

        for i, thr in enumerate(thresholds):
            predicted = (df['out'] > thr)
            tp = sum(predicted & df['label'])
            fp = sum(predicted & ~df['label'])
            fn = sum(~predicted & df['label'])
            tn = sum(~predicted & ~df['label'])

            fp_arr[i] = fp
            fn_arr[i] = fn

            # Evaluation metrics
            _, prec[i], rec[i], fpr[i], fscore[i] = get_metrics(tp,
                                                                fp,
                                                                tn,
                                                                fn,
                                                                args.beta)

        model_names.append(model_name)
        prec_list.append(prec)
        rec_list.append(rec)
        fpr_list.append(fpr)
        fscore_list.append(fscore)

        fp_list.append(fp_arr)
        fn_list.append(fn_arr)

    # Graficar

    # PR
    plt.figure(figsize=(12, 9))

    for i in range(2):
        plt.plot(rec_list[i], prec_list[i], '--o', label=model_names[i])

    plt.title("Comparación curva PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend([model_names[0], model_names[1]], loc='best')
    plt.savefig(f'Analysis/COMPARACION/PR.png')

    # ROC
    plt.figure(figsize=(12, 9))

    for i in range(2):
        plt.plot(fpr_list[i], rec_list[i][::-1], '--o', label=model_names[i])

    plt.title("Comparación curva ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.legend([model_names[0], model_names[1]], loc='best')
    plt.savefig(f'Analysis/COMPARACION/ROC.png')

    # FSCORE
    plt.figure(figsize=(12, 9))

    for i in range(2):
        plt.plot(thresholds, fscore_list[i], '--o', label=model_names[i])

    plt.title("Comparación curva Fscore")
    plt.xlabel("Threshold")
    plt.ylabel("Fscore")
    plt.grid(True)
    plt.legend([model_names[0], model_names[1]], loc='best')
    plt.savefig(f'Analysis/COMPARACION/Fscore.png')

    # FPFN
    plt.figure(figsize=(12, 9))

    for i in range(2):
        plt.plot(thresholds, fp_list[i], '--o', label=model_names[i] + '_fp')
        plt.plot(thresholds, fn_list[i], '--o', label=model_names[i] + '_fn')

    plt.title("Comparación curva FPFN")
    plt.xlabel("Threshold")
    plt.ylabel("Counts")
    plt.grid(True)
    plt.legend([model_names[0] + '_fp', model_names[0] + '_fn',
                model_names[1] + '_fp', model_names[1] + '_fn'],
               loc='best')

    plt.savefig(f'Analysis/COMPARACION/FPFN.png')


def get_pr_roc_auc(precision, recall, fpr):
    pr_auc = np.trapz(precision[::-1], x=recall[::-1])
    roc_auc = np.trapz(recall[::-1], x=fpr[::-1])

    return pr_auc, roc_auc


def save_fig(x, y, xlabel, ylabel, title, fname):
    plt.figure(figsize=(12, 9))
    plt.plot(x, y, '--o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(f'Analysis/' + fname)


def get_metrics(tp, fp, tn, fn, beta):
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
        fscore = (1 + beta ** 2) * (precision * recall) / \
                 ((beta ** 2) * precision + recall)

    return acc, recall, precision, fpr, fscore


def print_metrics(acc, recall, precision, fpr, fscore):
    print(f'Accuracy: {acc:5.3f}\n'
          f'Precision: {precision:5.3f}\n'
          f'Recall: {recall:5.3f}\n'
          f'False positive rate: {fpr:5.3f}\n'
          f'F-score: {fscore:5.3f}\n')

    # f'Total seismic traces: {tp + fn}\n'
    # f'Total non seismic traces: {tn + fp}\n'
    # f'correct: {tp + tn} / {tp + fp + tn + fn} \n\n'
    # f'True positives: {tp}\n'
    # f'True negatives: {tn}\n'
    # f'False positives: {fp}\n'
    # f'False negatives: {fn}\n\n'


if __name__ == "__main__":
    main()

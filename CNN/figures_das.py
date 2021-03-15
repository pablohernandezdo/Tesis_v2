import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file',
                        default='Results/Testing/Outputs/'
                                'DAS/separated/Cnn1_3k_10_1e4_256_40.csv',
                        help='Path to CSV file')
    parser.add_argument("--beta", type=float, default=2,
                        help="Fscore beta parameter")
    args = parser.parse_args()

    # Create figures files folder

    Path(f"Results/Testing/Fscore/DAS").mkdir(parents=True, exist_ok=True)

    Path(f"Results/Testing/Histogram/DAS").mkdir(parents=True, exist_ok=True)

    Path(f"Results/Testing/PR/DAS").mkdir(parents=True, exist_ok=True)

    Path(f"Results/Testing/ROC/DAS").mkdir(parents=True, exist_ok=True)

    # Define threshold to evaluate on
    thresholds = np.arange(0, 1, 0.01)

    # Read csv file
    df = pd.read_csv(args.csv_file)

    # Get model name
    model_name = args.csv_file.split('/')[-1].split('.')[0]

    # Preallocate variables
    acc = np.zeros(len(thresholds))
    prec = np.zeros(len(thresholds))
    rec = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    fscore = np.zeros(len(thresholds))

    for i, thr in enumerate(thresholds):
        predicted = (df['out'] > thr)
        tp = sum(predicted & df['label'])
        fp = sum(predicted & ~df['label'])
        fn = sum(~predicted & df['label'])
        tn = sum(~predicted & ~df['label'])

        # Evaluation metrics
        acc[i], prec[i], rec[i], fpr[i], fscore[i] = get_metrics(tp,
                                                                 fp,
                                                                 tn,
                                                                 fn,
                                                                 args.beta)

    # Obtain PR and ROC auc
    pr_auc, roc_auc = get_pr_roc_auc(prec, rec, fpr)

    # Get best threshold
    best_idx = np.argmax(fscore)
    print(thresholds[best_idx])

    # Histogram
    save_histogram(df, f'Results/Testing/Histogram/DAS/'
                       f'{model_name}_Histogram.png')

    # F-score vs thresholds curve
    save_fsc(thresholds, fscore,
             f'Results/Testing/Fscore/DAS/{model_name}_Fscore_vs_Threshold.png')

    # Precision vs recall (PR curve)
    save_pr(rec, prec,
            f'Results/Testing/PR/DAS/{model_name}_PR_curve.png')

    # Recall vs False Positive Rate (ROC curve)
    save_roc(fpr, rec[::-1],
             f'Results/Testing/ROC/DAS/{model_name}_ROC_curve.png')


def save_pr(recall, precision, figname):
    plt.figure(figsize=(12, 9))
    plt.plot(recall, precision, '--o')
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall (PR curve)')
    plt.grid(True)
    plt.savefig(figname)
    plt.close()


def save_roc(fpr, recall, figname):
    plt.figure(figsize=(12, 9))
    plt.plot(fpr, recall, '--o')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('Recall vs FPR (ROC curve)')
    plt.grid(True)
    plt.savefig(figname)
    plt.close()


def save_fsc(thresholds, fscore, figname):
    plt.figure(figsize=(12, 9))
    plt.plot(thresholds, fscore, '--o')
    plt.xlabel('Thresholds')
    plt.ylabel('F-score')
    plt.title('Fscores vs Thresholds')
    plt.grid(True)
    plt.savefig(figname)
    plt.close()


def save_histogram(df, hist_name):
    plt.figure(figsize=(12, 9))
    plt.hist(df[df['label'] == 1]['out'], 100)
    plt.hist(df[df['label'] == 0]['out'], 100)
    plt.title(f'Output values histogram')
    plt.xlabel('Output values')
    plt.ylabel('Counts')
    plt.legend(['positive', 'negative'], loc='upper left')
    plt.grid(True)
    plt.savefig(hist_name)


def get_pr_roc_auc(precision, recall, fpr):
    pr_auc = np.trapz(precision[::-1], x=recall[::-1])
    roc_auc = np.trapz(recall[::-1], x=fpr[::-1])

    return pr_auc, roc_auc


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

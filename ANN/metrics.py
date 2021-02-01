# file metrics.py
# brief Script to obtain & plot statistics metrics of a CNN model
# author Jaime Ramirez
# date Ene, 2021

import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder', help='Path to CSV files folder')
    parser.add_argument("--beta",
                        type=float,
                        default=2,
                        help="Fscore beta parameter")
    args = parser.parse_args()

    # Folder to save files
    folder2save = args.csv_folder.split("/")[-2] + '/' + args.csv_folder.split("/")[-1]

    # Create csv files folder
    Path(f"../Analysis/FigsCSV/{folder2save}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/FigsCSV/{folder2save}").mkdir(parents=True, exist_ok=True)

    thresholds = np.arange(0, 1, 0.01)

    # Files from folder
    for csv in os.listdir(args.csv_folder):

        model_name = csv.strip().split('.')[0]
        df = pd.read_csv(f'{args.csv_folder}/{csv}')

        precision = np.zeros(len(thresholds))
        recall = np.zeros(len(thresholds))
        fpr = np.zeros(len(thresholds))
        fscore = np.zeros(len(thresholds))

        for i, thr in enumerate(thresholds):
            predicted = (df['out'] > thr)
            tp = sum(predicted & df['label'])
            fp = sum(predicted & ~df['label'])
            fn = sum(~predicted & df['label'])
            tn = sum(~predicted & ~df['label'])

            # Evaluation metrics
            precision[i], recall[i], fpr[i], fscore[i] = get_metrics(tp,
                                                                     fp,
                                                                     tn,
                                                                     fn,
                                                                     args.beta)

        # Output histogram
        plt.figure(figsize=(12, 9))
        plt.hist(df[df['label'] == 1]['out'], 100)
        plt.hist(df[df['label'] == 0]['out'], 100)
        plt.title(f'{model_name} output histogram')
        plt.xlabel('Output values')
        plt.ylabel('Counts')
        plt.legend(['positive', 'negative'], loc='upper left')
        plt.grid(True)
        plt.savefig(f'{} + model_name + '_Histogram.png')

        # F-score vs thresholds curve
        save_fig(thresholds,
                 fscore,
                 'Threshold',
                 'F-score',
                 'Fscores vs Thresholds',
                 f'{folder2save}/{model_name}_Fscore_vs_Threshold.png')

        # Precision vs recall (PR curve)
        save_fig(recall,
                 precision,
                 'Recall',
                 'Precision',
                 'Precision vs Recall (PR curve)',
                 f'folder2save}/{model_name}_PR_curve.png')

        # Recall vs False Positive Rate (ROC curve)
        save_fig(fpr,
                 recall,
                 'False Positive Rate',
                 'Recall',
                 'Recall vs FPR (ROC curve)',
                 f'folder2save}/{model_name}_ROC_curve.png')


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
        fscore = (1 + beta**2) * (precision * recall) / ((beta**2)*precision + recall)

    return precision, recall, fpr, fscore


def save_fig(x, y, xlabel, ylabel, title, fname):
    plt.figure(figsize=(12, 9))
    plt.plot(x, y, '--o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(f'../Analysis/FigsCSV/' + fname)


if __name__ == "__main__":
    main()

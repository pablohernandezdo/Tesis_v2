import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder',
                        default='Results/Testing/Outputs',
                        help='Path to CSV files folder')
    parser.add_argument('--model_name', default='Cnn1_3k_10_1e4_256_40',
                        help='Path to CSV files folder')
    parser.add_argument("--beta", type=float, default=2,
                        help="Fscore beta parameter")
    args = parser.parse_args()

    # Define threshold to evaluate on
    thresholds = np.arange(0, 1, 0.01)

    dataset_names = ["Stead_seismic_test",
                     "Stead_noise_test",
                     "Geo_test"]

    # Preallocate variables
    acc = np.zeros(len(thresholds))
    prec = np.zeros(len(thresholds))
    rec = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    fscore = np.zeros(len(thresholds))

    thr_test = 60

    for i, thr in enumerate(thresholds):

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for dset in dataset_names:
            # leer los csv de cada dataset, obtener los tp, fp, fn, tn
            df = pd.read_csv(f'{args.csv_folder}/{dset}/'
                             f'separated/{args.model_name}.csv')

            predicted = (df['out'] > thr)
            tp += sum(predicted & df['label'])
            fp += sum(predicted & ~df['label'])
            fn += sum(~predicted & df['label'])
            tn += sum(~predicted & ~df['label'])

            if i == thr_test:
                print(f'dataset: {dset}\n'
                      f"tp: {sum(predicted & df['label'])},"
                      f" fp: {sum(predicted & ~df['label'])},"
                      f" fn: {sum(~predicted & df['label'])},"
                      f" tn: {sum(~predicted & ~df['label'])}")

        # Evaluation metrics
        acc[i], prec[i], rec[i], fpr[i], fscore[i] = get_metrics(tp,
                                                                 fp,
                                                                 tn,
                                                                 fn,
                                                                 args.beta)

    print(f'thr_test: {thresholds[thr_test]}'
          f'fscore: {fscore[thr_test]}')

    # best_idx = np.argmax(fscore)
    # best_fsc = np.amax(fscore)
    # best_thresh = thresholds[best_idx]
    #
    # print(f'best_idx: {best_idx}\n'
    #       f'best_fscore: {best_fsc}\n'
    #       f'best_thresh: {best_thresh}')


    Path(f"Results/Testing/Histogram/"
         f"separated/").mkdir(parents=True, exist_ok=True)

    Path(f"Results/Testing/Fscore/"
         f"separated/").mkdir(parents=True, exist_ok=True)

    Path(f"Results/Testing/PR/"
         f"separated/").mkdir(parents=True, exist_ok=True)

    Path(f"Results/Testing/ROC/"
         f"separated/").mkdir(parents=True, exist_ok=True)

    # F-score vs thresholds curve
    save_fig(thresholds,
             fscore,
             'Threshold',
             'F-score',
             'Fscores vs Thresholds',
             f'Results/Testing/Fscore/separated/'
             f'{args.model_name}_Fscore_vs_Threshold.png')

    # Precision vs recall (PR curve)
    save_fig(rec,
             prec,
             'Recall',
             'Precision',
             'Precision vs Recall (PR curve)',
             f'Results/Testing/PR/separated/'
             f'{args.model_name}_PR_curve.png')

    # Recall vs False Positive Rate (ROC curve)
    save_fig(fpr,
             rec[::-1],
             'False Positive Rate',
             'Recall',
             'Recall vs FPR (ROC curve)',
             f'Results/Testing/ROC/separated/'
             f'{args.model_name}_ROC_curve.png')

    plt.close('all')


def get_model_figures(thresholds, fscore, recall, precision, fpr,
                      figure_folders_path, dataset, model_name):

    # Get best threshold
    best_idx = np.argmax(fscore)

    # F-score vs thresholds curve
    save_fig(thresholds,
             fscore,
             'Threshold',
             'F-score',
             'Fscores vs Thresholds',
             f'{figure_folders_path}/Fscore/'
             f'{dataset}/{model_name}_Fscore_vs_Threshold.png')

    # Precision vs recall (PR curve)
    save_fig(recall,
             precision,
             'Recall',
             'Precision',
             'Precision vs Recall (PR curve)',
             f'{figure_folders_path}/PR/'
             f'{dataset}/{model_name}_PR_curve.png')

    # Recall vs False Positive Rate (ROC curve)
    save_fig(fpr,
             recall[::-1],
             'False Positive Rate',
             'Recall',
             'Recall vs FPR (ROC curve)',
             f'{figure_folders_path}/ROC/'
             f'{dataset}/{model_name}_ROC_curve.png')

    plt.close('all')


def get_folder_figures():
    pass


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


def save_fig(x, y, xlabel, ylabel, title, fname):
    plt.figure(figsize=(12, 9))
    plt.plot(x, y, '--o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(fname)


if __name__ == "__main__":
    main()
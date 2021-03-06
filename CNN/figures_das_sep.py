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

    Path(f"Results/Testing/Fscore/"
         f"DAS_sep/").mkdir(parents=True, exist_ok=True)

    Path(f"Results/Testing/PR/"
         f"DAS_sep/").mkdir(parents=True, exist_ok=True)

    Path(f"Results/Testing/ROC/"
         f"DAS_sep/").mkdir(parents=True, exist_ok=True)

    # Define threshold to evaluate on
    thresholds = np.arange(0, 1, 0.01)

    dataset_names = ["DAS_seismic",
                     "DAS_non_seismic",
                     "DAS_noise"]

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

            Path(f"Results/Testing/Histogram/"
                 f"DAS_sep/{dset}").mkdir(parents=True, exist_ok=True)

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

    for dset in dataset_names:

        # leer los csv de cada dataset
        df = pd.read_csv(f'{args.csv_folder}/{dset}/'
                         f'separated/{args.model_name}.csv')

        # Guardar histograma
        plt.figure(figsize=(12, 9))
        plt.hist(df[df['label'] == 1]['out'], 100)
        plt.hist(df[df['label'] == 0]['out'], 100)
        plt.title(f'{args.model_name}, {dset} output histogram')
        plt.xlabel('Output values')
        plt.ylabel('Counts')
        plt.legend(['positive', 'negative'], loc='upper left')
        plt.grid(True)
        plt.savefig(f'Results/Testing/Histogram/'
                    f'DAS_sep/{dset}/{args.model_name}_Histogram.png')

        # Guardar histograma eje y log
        plt.figure(figsize=(12, 9))
        plt.hist(df[df['label'] == 1]['out'], 100)
        plt.hist(df[df['label'] == 0]['out'], 100)
        plt.yscale('log')
        plt.title(f'{args.model_name}, {dset} output histogram')
        plt.xlabel('Output values')
        plt.ylabel('Counts')
        plt.legend(['positive', 'negative'], loc='upper left')
        plt.grid(True)
        plt.savefig(f'Results/Testing/Histogram/'
                    f'DAS_sep/{dset}/{args.model_name}_log_Histogram.png')


    print(f'thr_test: {thresholds[thr_test]}\n'
          f'fscore: {fscore[thr_test]}')

    best_idx = np.argmax(fscore)
    best_fsc = np.amax(fscore)
    best_thresh = thresholds[best_idx]
    print(f'best_thr: {best_thresh}')
    print(f'best_fscore: {best_fsc}')

    # print(f'best_idx: {best_idx}\n'
    #       f'best_fscore: {best_fsc}\n'
    #       f'best_thresh: {best_thresh}')

    # F-score vs thresholds curve
    save_fsc(thresholds, fscore,
             f'Results/Testing/Fscore/DAS_sep/{args.model_name}_Fscore_vs_Threshold.png')

    # Precision vs recall (PR curve)
    save_pr(rec, prec,
            f'Results/Testing/PR/DAS_sep/{args.model_name}_PR_curve.png')

    # Recall vs False Positive Rate (ROC curve)
    save_roc(fpr, rec[::-1],
             f'Results/Testing/ROC/DAS_sep/{args.model_name}_ROC_curve.png')


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


def get_model_figures(thresholds, fscore, recall, precision, fpr,
                      figure_folders_path, dataset, model_name):

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
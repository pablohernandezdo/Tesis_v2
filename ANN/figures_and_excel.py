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
                        help='Path to CSV files folder')
    parser.add_argument('--xls_name',
                        help='Best models excel file name')
    parser.add_argument("--beta", type=float, default=2,
                        help="Fscore beta parameter")
    args = parser.parse_args()

    # Folder to save files
    folder2save = f'{args.csv_folder.split("/")[-2]}/' \
                  f'{args.csv_folder.split("/")[-1]}'

    # Create figures and excel files folder
    Path("../Analysis/Excel_reports").mkdir(parents=True, exist_ok=True)

    Path(f"../Analysis/FigsCSV/Fscore/"
         f"{folder2save}").mkdir(parents=True, exist_ok=True)

    Path(f"../Analysis/FigsCSV/Histogram/"
         f"{folder2save}").mkdir(parents=True, exist_ok=True)

    Path(f"../Analysis/FigsCSV/PR/"
         f"{folder2save}").mkdir(parents=True, exist_ok=True)

    Path(f"../Analysis/FigsCSV/ROC/"
         f"{folder2save}").mkdir(parents=True, exist_ok=True)

    # Define threshold to evaluate on
    thresholds = np.arange(0, 1, 0.01)

    # Models best values

    models = []
    best_acc = []
    best_prec = []
    best_rec = []
    best_fpr = []
    best_fscore = []
    best_pr_auc = []
    best_roc_auc = []

    # Read Csv files
    for csv in os.listdir(args.csv_folder):

        # Read csv file
        df = pd.read_csv(f'{args.csv_folder}/{csv}')

        # Get model name
        model_name = os.path.splitext(csv)[0]

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

        # Add best threshold values to best models list
        models.append(model_name)
        best_acc.append(acc[best_idx])
        best_prec.append(prec[best_idx])
        best_rec.append(rec[best_idx])
        best_fpr.append(fpr[best_idx])
        best_fscore.append(fscore[best_idx])
        best_pr_auc.append(pr_auc)
        best_roc_auc.append(roc_auc)

        # Guardar graficas

        # Output histogram
        plt.figure(figsize=(12, 9))
        plt.hist(df[df['label'] == 1]['out'], 100)
        plt.hist(df[df['label'] == 0]['out'], 100)
        plt.title(f'{model_name} output histogram')
        plt.xlabel('Output values')
        plt.ylabel('Counts')
        plt.legend(['positive', 'negative'], loc='upper left')
        plt.grid(True)
        plt.savefig(f'../Analysis/FigsCSV/Histogram/'
                    f'{folder2save}/{model_name}_Histogram.png')

        # F-score vs thresholds curve
        save_fig(thresholds,
                 fscore,
                 'Threshold',
                 'F-score',
                 'Fscores vs Thresholds',
                 f'Fscore/{folder2save}/{model_name}_Fscore_vs_Threshold.png')

        # Precision vs recall (PR curve)
        save_fig(rec,
                 prec,
                 'Recall',
                 'Precision',
                 'Precision vs Recall (PR curve)',
                 f'PR/{folder2save}/{model_name}_PR_curve.png')

        # Recall vs False Positive Rate (ROC curve)
        save_fig(fpr,
                 rec,
                 'False Positive Rate',
                 'Recall',
                 'Recall vs FPR (ROC curve)',
                 f'ROC/{folder2save}/{model_name}_ROC_curve.png')

        plt.close('all')

    # Save csv with best models

    print(len(models))

    # Get indexes to sort by fscore
    sorted_idxs = np.argsort(best_fscore)

    # Sorted lists
    sorted_models = [models[i] for i in sorted_idxs[::-1]]
    sorted_acc = [best_acc[i] for i in sorted_idxs[::-1]]
    sorted_prec = [best_prec[i] for i in sorted_idxs[::-1]]
    sorted_rec = [best_rec[i] for i in sorted_idxs[::-1]]
    sorted_fpr = [best_fpr[i] for i in sorted_idxs[::-1]]
    sorted_fscore = [best_fscore[i] for i in sorted_idxs[::-1]]
    sorted_pr_auc = [best_pr_auc[i] for i in sorted_idxs[::-1]]
    sorted_roc_auc = [best_roc_auc[i] for i in sorted_idxs[::-1]]

    # Create dataframe
    df = pd.DataFrame({
        'Model_name': sorted_models,
        'Accuracy': sorted_acc,
        'Precision': sorted_prec,
        'Recall': sorted_rec,
        'False positive rate': sorted_fpr,
        'Fscore': sorted_fscore,
        'PR AUC': sorted_pr_auc,
        'ROC_AUC': sorted_roc_auc,
    })

    # Save excel file
    with pd.ExcelWriter(f'../Analysis/Excel_reports/{args.xls_name}.xlsx',
                        engine='openpyxl') as writer:

        df.to_excel(writer, index=False)


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
    plt.savefig(f'../Analysis/FigsCSV/' + fname)


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

import os
import argparse

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Create folder for report
    Path("../Analysis/Excel_reports").mkdir(exist_ok=True)
    Path("../Analysis/Curves_parameters").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--best_folder', default='default', help='Best models log folder')
    parser.add_argument('--avg_folder', default='default', help='Average models log folder')
    parser.add_argument('--best_models', default='', help='Best model names space separated')
    parser.add_argument('--avg_models', default='', help='Average model names space separated')
    parser.add_argument('--n_thresh', type=int, default=18, help='Number of thresholds evaluated')
    args = parser.parse_args()

    # working directory
    best_eval_wkdir = '../Analysis/logs/eval/' + args.best_folder
    avg_eval_wkdir = '../Analysis/logs/eval/' + args.avg_folder

    best_models = args.best_models.strip().split(' ')
    avg_models = args.avg_models.strip().split(' ')

    # Variable preallocating
    thresholds = []

    pre = []
    rec = []
    fpr = []
    fsc = []

    pr_curves = []
    roc_curves = []
    fscore_curves = []

    pr_aucs = []
    roc_aucs = []

    avg_pre = []
    avg_rec = []
    avg_fpr = []
    avg_fsc = []

    avg_pr_curves = []
    avg_roc_curves = []
    avg_fscore_curves = []

    avg_pr_aucs = []
    avg_roc_aucs = []

    for f_name in avg_models:
        with open(os.path.join(avg_eval_wkdir, f_name), 'r') as f:

            f.readline()
            f.readline()

            for _ in range(args.n_thresh):

                thresh = f.readline().split(':')[-1].strip()
                thresholds.append(thresh)

                # Skip non-useful lines
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # acc
                f.readline()

                # Read metrics
                avg_pre.append(f.readline().split(":")[-1].strip())
                avg_rec.append(f.readline().split(":")[-1].strip())
                avg_fpr.append(f.readline().split(":")[-1].strip())
                avg_fsc.append(f.readline().split(":")[-1].strip())

                f.readline()
                f.readline()
                f.readline()

            # Terminar de leer el archivo
            best_fscore = f.readline().split(",")[-1].strip().split(":")[-1].strip()

            f.readline()

            avg_pr_aucs.append(f.readline().split(":")[-1].strip())
            avg_roc_aucs.append(f.readline().split(":")[-1].strip())

            # print(f'rec: {rec}\npre: {pre}\nfpr: {fpr}\nfsc: {fsc}')

            avg_pre = list(map(float, avg_pre))
            avg_rec = list(map(float, avg_rec))
            avg_fpr = list(map(float, avg_fpr))
            avg_fsc = list(map(float, avg_fsc))
            thresholds = list(map(float, thresholds))

            # Aqui armar la curva y agregarlas a la lista mayor
            avg_pr_curves.append([avg_rec, avg_pre])
            avg_roc_curves.append([avg_fpr, avg_rec])
            avg_fscore_curves.append([thresholds, avg_fsc])

        avg_pre = []
        avg_rec = []
        avg_fpr = []
        avg_fsc = []
        thresholds = []

    for f_name in best_models:
        with open(os.path.join(best_eval_wkdir, f_name), 'r') as f:

            f.readline()
            f.readline()

            for _ in range(args.n_thresh):

                thresh = f.readline().split(':')[-1].strip()
                thresholds.append(thresh)

                # Skip non-useful lines
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # acc
                f.readline()

                # Read metrics
                pre.append(f.readline().split(":")[-1].strip())
                rec.append(f.readline().split(":")[-1].strip())
                fpr.append(f.readline().split(":")[-1].strip())
                fsc.append(f.readline().split(":")[-1].strip())

                f.readline()
                f.readline()
                f.readline()

            # Terminar de leer el archivo
            best_fscore = f.readline().split(",")[-1].strip().split(":")[-1].strip()

            f.readline()

            pr_aucs.append(f.readline().split(":")[-1].strip())
            roc_aucs.append(f.readline().split(":")[-1].strip())

            # print(f'rec: {rec}\npre: {pre}\nfpr: {fpr}\nfsc: {fsc}')

            pre = list(map(float, pre))
            rec = list(map(float, rec))
            fpr = list(map(float, fpr))
            fsc = list(map(float, fsc))
            thresholds = list(map(float, thresholds))

            # Aqui armar la curva y agregarlas a la lista mayor
            pr_curves.append([rec, pre])
            roc_curves.append([fpr, rec])
            fscore_curves.append([thresholds, fsc])

        pre = []
        rec = []
        fpr = []
        fsc = []
        thresholds = []

    # Curvas PR
    plt.figure()

    for crv in avg_pr_curves:
        plt.plot(crv[0], crv[1])

    for crv in pr_curves:
        plt.plot(crv[0], crv[1])

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'PR curves best models LSTM')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.02, 1.02)
    plt.ylim(0.48, 1.02)
    plt.grid(True)
    plt.savefig(f'../Analysis/Curves_parameters/best_PR_lstm.png')

    # Curva ROC
    plt.figure()

    for crv in avg_roc_curves:
        plt.plot(crv[0], crv[1])

    for crv in roc_curves:
        plt.plot(crv[0], crv[1])

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curves best models LSTM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.savefig(f'../Analysis/Curves_parameters/best_ROC_lstm.png')

    # Curva Fscore
    plt.figure()

    for crv in avg_fscore_curves:
        plt.plot(crv[0], crv[1])

    for crv in fscore_curves:
        plt.plot(crv[0], crv[1])

    plt.title(f'Fscore vs thresholds curves best models LSTM')
    plt.xlabel('Threshold')
    plt.ylabel('F-score')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.savefig(f'../Analysis/Curves_parameters/best_Fscore_lstm.png')


if __name__ == "__main__":
    main()

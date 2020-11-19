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
    parser.add_argument('--archives_folder', default='default', help='Name of excel file to export')
    parser.add_argument('--best_models', default='', help='Best model names space separated')
    parser.add_argument('--n_thresh', type=int, default=18, help='Number of thresholds evaluated')
    args = parser.parse_args()

    # working directory
    eval_wkdir = '../Analysis/logs/eval/' + args.archives_folder

    models = args.best_models.strip().split(' ')

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

    for f_name in models:
        with open(os.path.join(eval_wkdir, f_name), 'r') as f:

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

            # Aqui armar la curva y agregarlas a la lista mayor
            pr_curves.append([rec, pre])
            roc_curves.append([fpr, rec])
            fscore_curves.append([thresholds, fsc])

        pre = []
        rec = []
        fpr = []
        fsc = []
        thresholds = []

    # Str to float

    # Test PR
    plt.figure()

    for crv in pr_curves:
        print(f'rec: {crv[0]}\npre: {crv[1]}')
        plt.plot(crv[0], crv[1])
        break

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'Test PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.02, 1.02)
    plt.ylim(0.48, 1.02)
    plt.grid(True)
    plt.savefig(f'../Analysis/Curves_parameters/test_PR_ann.png')

    # # Curvas PR
    # plt.figure()
    #
    # for crv in pr_curves:
    #     plt.plot(crv[0], crv[1])
    #
    # # Dumb model line
    # plt.hlines(0.5, 0, 1, 'b', '--')
    # plt.title(f'PR curves best models ANN')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.xlim(-0.02, 1.02)
    # plt.ylim(0.48, 1.02)
    # plt.grid(True)
    # plt.savefig(f'../Analysis/Curves_parameters/best_PR_ann.png')
    #
    # # Curva ROC
    # plt.figure()
    #
    # for crv in roc_curves:
    #     plt.plot(crv[0], crv[1])
    #
    # # Dumb model line
    # plt.plot([0, 1], [0, 1], 'b--')
    # plt.title(f'ROC curves best models ANN')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('Recall')
    # plt.xlim(-0.02, 1.02)
    # plt.ylim(-0.02, 1.02)
    # plt.grid(True)
    # plt.savefig(f'../Analysis/Curves_parameters/best_ROC_ann.png')
    #
    # # Curva Fscore
    # plt.figure()
    #
    # for crv in fscore_curves:
    #     plt.plot(crv[0], crv[1])
    #
    # plt.title(f'Fscore vs thresholds curves best models ANN')
    # plt.xlabel('Threshold')
    # plt.ylabel('F-score')
    # plt.xlim(-0.02, 1.02)
    # plt.ylim(-0.02, 1.02)
    # plt.grid(True)
    # plt.savefig(f'../Analysis/Curves_parameters/best_Fscore_ann.png')


if __name__ == "__main__":
    main()

import os
import argparse

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Create folder for report
    Path("Analysis/Excel_reports").mkdir(exist_ok=True)
    Path("Analysis/Curves_parameters").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--archives_folder', default='default', help='Name of excel file to export')
    parser.add_argument('--best_models', default='', help='Best model names space separated')
    parser.add_argument('--n_thresh', type=int, default=18, help='Number of thresholds evaluated')
    args = parser.parse_args()

    # working directory
    eval_wkdir = 'Analysis/logs/eval/' + args.archives_folder

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
        with open(os.path.join(eval_wkdir, models), 'r') as f:

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

            # Aqui armar la curva y agregarlas a la lista mayor
            pr_curves.append([pre, rec])
            roc_curves.append([fpr, rec,])
            fscore_curves.append([fsc, thresholds])

        pre = []
        rec = []
        fpr = []
        fsc = []
        thresholds = []

    # Aqui graficar todas las curvas y guardarlas en una carpeta


if __name__ == "__main__":
    main()

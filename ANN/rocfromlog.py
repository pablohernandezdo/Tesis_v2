import os
import argparse

from pathlib import Path

import numpy as np


def main():
    # Create folder for report
    Path("ROCfromLog").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='2h1k5k_5e5_256_20', help='Name of model to obtain curves')
    parser.add_argument('--archives_folder', default='step4_curves', help='Name of excel file to export')
    parser.add_argument('--n_thresh', type=int, default=19, help='Number of thresholds evaluated')
    args = parser.parse_args()

    # working directory
    wkdir = 'logs/eval/' + args.archives_folder

    # Variable preallocating
    pre = []
    rec = []
    fpr = []
    fsc = []

    # Ruta del archivo
    model_file = os.path.join(wkdir, args.model_name)

    # Leer archivo
    with open(model_file + '.txt', 'r') as f:

        # Skip initial lines
        f.readline()
        f.readline()

        for _ in range(args.n_thresh):

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
            f.readline()
            f.readline()

            pre.append(f.readline().split(":")[-1].strip())
            rec.append(f.readline().split(":")[-1].strip())
            fpr.append(f.readline().split(":")[-1].strip())
            fsc.append(f.readline().split(":")[-1].strip())

            f.readline()
            f.readline()
            f.readline()

        f.readline()
        f.readline()

        pr_auc = f.readline().split(":")[-1].strip()
        roc_auc = f.readline().split(":")[-1].strip()


    print(f'precision: {pre}')
    print(f'recall: {rec}')
    print(f'fpr: {fpr}')

    # Add point (0, 1) to PR curve
    pre.append(1.0)
    rec.append(0.0)

    # Add point (1, 0.5) to PR curve
    pre.insert(0, 0.5)
    rec.insert(0, 1.0)

    # Add point (0, 0)  to ROC curve
    fpr.append(0.0)

    # Add point (1, 1) to ROC curve
    fpr.insert(0, 1.0)

    print(f'precision: {pre}')
    print(f'recall: {rec}')
    print(f'fpr: {fpr}')

    pre = np.array(pre)
    rec = np.array(rec)
    fpr = np.array(fpr)
    
    # Area under curve
    calc_pr_auc = np.trapz(pre, x=rec[::-1])
    calc_roc_auc = np.trapz(rec, x=fpr[::-1])


if __name__ == "__main__":
    main()

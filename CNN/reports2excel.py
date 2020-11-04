import os
import argparse

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # Create folder for report
    Path("../Excel_reports").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', default='eval', help='Name of folder to read log files')
    parser.add_argument('--xls_name', default='train_xls', help='Name of excel file to export')
    parser.add_argument('--n_thresh', type=int, default=19, help='Number of thresholds evaluated')
    args = parser.parse_args()

    # working directory
    wkdir = os.path.join('../logs', args.folder_name)

    # Variable preallocating
    models = []
    params = []
    thresholds = []

    tp = []
    tn = []
    fp = []
    fn = []

    acc = []
    pre = []
    rec = []
    fpr = []
    fsc = []

    ev_tm = []

    # Obtener los archivos de la carpeta
    files = os.listdir(wkdir)

    # Leer los archivos en la carpeta
    for fname in files:
        with open(os.path.join(wkdir, fname), 'r') as f:
            model_name = fname.split('.')[0]

            # Skip initial empty lines
            model_params = f.readline().split(":")[-1].strip()
            f.readline()

            # Start reading threshold data
            for _ in range(args.n_thresh):
                models.append(model_name)
                params.append(model_params)

                thresh = f.readline().split(':')[-1].strip()
                thresholds.append(thresh)

                # Skip non-useful lines
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # Read 4 cases
                tp.append(f.readline().split(":")[-1].strip())
                tn.append(f.readline().split(":")[-1].strip())
                fp.append(f.readline().split(":")[-1].strip())
                fn.append(f.readline().split(":")[-1].strip())

                # Skip empty line
                f.readline()

                # Read metrics
                acc.append(f.readline().split(":")[-1].strip())
                pre.append(f.readline().split(":")[-1].strip())
                rec.append(f.readline().split(":")[-1].strip())
                fpr.append(f.readline().split(":")[-1].strip())
                fsc.append(f.readline().split(":")[-1].strip())

                # Skip empty line
                f.readline()

                # Read eval time
                ev_tm.append(f.readline().split(":")[-1].strip())

                # Skip empty line
                f.readline()

            # Read final report
            print(f'best thresh = {f.readline().split(":")[-2].split(",")[0].strip()}')
            print(f'best fscore = {f.readline().split(":")[-1].strip()}')

            # Skip empty line
            f.readline()

            print(f'PR AUC = {f.readline().split(":")[-1].strip()}')
            print(f'ROC AUC = {f.readline().split(":")[-1].strip()}')

    df = pd.DataFrame({
        'Model_name': models,
        'Parameters': params,
        'Threshold': thresholds,
        'Evaluation time': ev_tm,
        'True positives': tp,
        'True negatives': tn,
        'False positives': fp,
        'False negatives': fn,
        'Accuracy': acc,
        'Precision': pre,
        'Recall': rec,
        'False positive rate': fpr,
        'F-score': fsc,
    })

    df.to_excel(f'../Excel_reports/{args.xls_name}.xlsx', index=False)


if __name__ == "__main__":
    main()

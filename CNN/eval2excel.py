import os
import argparse

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # Create folder for report
    Path("../Excel_reports").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--xls_name', default='eval_xls', help='Name of excel file to export')
    args = parser.parse_args()

    # working directory
    train_wkdir = '../logs/train'
    eval_wkdir = '../logs/eval'

    # Variable preallocating
    models = []
    params = []

    tr_time = []

    tr_tp = []
    tr_tn = []
    tr_fp = []
    tr_fn = []

    tr_acc = []
    tr_pre = []
    tr_rec = []
    tr_fpr = []
    tr_fsc = []

    tr_ev_tm = []

    tst_tp = []
    tst_tn = []
    tst_fp = []
    tst_fn = []

    tst_acc = []
    tst_pre = []
    tst_rec = []
    tst_fpr = []
    tst_fsc = []

    tst_ev_tm = []

    # Obtener los archivos de la carpeta
    train_files = os.listdir(train_wkdir)
    eval_files = os.listdir(eval_wkdir)

    # Leer tiempo de entrenamiento
    for fname in train_files:
        with open(os.path.join(train_wkdir, fname), 'r') as f:

            f.readline()
            f.readline()
            f.readline()

            tr_time.append(f.readline().split(":")[-1].strip())

    # Leer los archivos en la carpeta
    for fname in eval_files:
        with open(os.path.join(eval_wkdir, fname), 'r') as f:
            model_name = fname.split('.')[0]
            models.append(model_name)

            # Skip initial empty lines
            f.readline()
            f.readline()
            f.readline()
            f.readline()

            tr_tp.append(f.readline().split(":")[-1].strip())
            tr_tn.append(f.readline().split(":")[-1].strip())
            tr_fp.append(f.readline().split(":")[-1].strip())
            tr_fn.append(f.readline().split(":")[-1].strip())

            f.readline()

            tr_acc.append(f.readline().split(":")[-1].strip())
            tr_pre.append(f.readline().split(":")[-1].strip())
            tr_rec.append(f.readline().split(":")[-1].strip())
            tr_fpr.append(f.readline().split(":")[-1].strip())
            tr_fsc.append(f.readline().split(":")[-1].strip())

            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()

            tst_tp.append(f.readline().split(":")[-1].strip())
            tst_tn.append(f.readline().split(":")[-1].strip())
            tst_fp.append(f.readline().split(":")[-1].strip())
            tst_fn.append(f.readline().split(":")[-1].strip())

            f.readline()

            tst_acc.append(f.readline().split(":")[-1].strip())
            tst_pre.append(f.readline().split(":")[-1].strip())
            tst_rec.append(f.readline().split(":")[-1].strip())
            tst_fpr.append(f.readline().split(":")[-1].strip())
            tst_fsc.append(f.readline().split(":")[-1].strip())

            f.readline()

            tr_ev_tm.append(f.readline().split(":")[-1].strip())
            tst_ev_tm.append(f.readline().split(":")[-1].strip())

            f.readline()
            f.readline()
            f.readline()

            params.append(f.readline().split(":")[-1].strip())

    df = pd.DataFrame({
        'Model_name': models,
        'Parameters': params,
        'Training time': tr_time,
        'Train Evaluation time': tr_ev_tm,
        'Train True positives': tr_tp,
        'Train True negatives': tr_tn,
        'Train False positives': tr_fp,
        'Train False negatives': tr_fn,
        'Train Accuracy': tr_acc,
        'Train Precision': tr_pre,
        'Train Recall': tr_rec,
        'Train False positive rate': tr_fpr,
        'Train F-score': tr_fsc,
        'Test Evaluation time': tst_ev_tm,
        'Test True positives': tst_tp,
        'Test True negatives': tst_tn,
        'Test False positives': tst_fp,
        'Test False negatives': tst_fn,
        'Test Accuracy': tst_acc,
        'Test Precision': tst_pre,
        'Test Recall': tst_rec,
        'Test False positive rate': tst_fpr,
        'Test F-score': tst_fsc,
    })

    df.to_excel(f'../Excel_reports/{args.xls_name}.xlsx', index=False)


if __name__ == "__main__":
    main()

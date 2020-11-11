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
    parser.add_argument('--archives_folder', default='default', help='Name of excel file to export')
    parser.add_argument('--best', type=int, default=10, help='Number of best models to save report')
    args = parser.parse_args()

    # working directory
    train_wkdir = '../logs/train/' + args.archives_folder
    eval_wkdir = '../logs/eval/' + args.archives_folder

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

            params.append(f.readline().split(":")[-1].strip())

    # Get the 10 highest F-score models
    best_idx = np.argsort(tst_fsc)

    # Get general parameters
    best_models = [models[i] for i in best_idx[:args.best]]
    best_params = [params[i] for i in best_idx[:args.best]]
    best_tr_time = [tr_time[i] for i in best_idx[:args.best]]
    best_tr_ev_tm = [tr_ev_tm[i] for i in best_idx[:args.best]]

    # Get train parameters
    best_tr_tp = [tr_tp[i] for i in best_idx[:args.best]]
    best_tr_tn = [tr_tn[i] for i in best_idx[:args.best]]
    best_tr_fp = [tr_fp[i] for i in best_idx[:args.best]]
    best_tr_fn = [tr_fn[i] for i in best_idx[:args.best]]
    best_tr_acc = [tr_acc[i] for i in best_idx[:args.best]]
    best_tr_pre = [tr_pre[i] for i in best_idx[:args.best]]
    best_tr_rec = [tr_rec[i] for i in best_idx[:args.best]]
    best_tr_fpr = [tr_fpr[i] for i in best_idx[:args.best]]
    best_tr_fsc = [tr_fsc[i] for i in best_idx[:args.best]]

    # Get test parameters
    best_tst_ev_tm = [tst_ev_tm[i] for i in best_idx[:args.best]]
    best_tst_tp = [tst_tp[i] for i in best_idx[:args.best]]
    best_tst_tn = [tst_tn[i] for i in best_idx[:args.best]]
    best_tst_fp = [tst_fp[i] for i in best_idx[:args.best]]
    best_tst_fn = [tst_fn[i] for i in best_idx[:args.best]]
    best_tst_acc = [tst_acc[i] for i in best_idx[:args.best]]
    best_tst_pre = [tst_pre[i] for i in best_idx[:args.best]]
    best_tst_rec = [tst_rec[i] for i in best_idx[:args.best]]
    best_tst_fpr = [tst_fpr[i] for i in best_idx[:args.best]]
    best_tst_fsc = [tst_fsc[i] for i in best_idx[:args.best]]

    df1 = pd.DataFrame({
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

    df2 = pd.DataFrame({
        'Model_name': best_models,
        'Parameters': best_params,
        'Training time': best_tr_time,
        'Train Evaluation time': best_tr_ev_tm,
        'Train True positives': best_tr_tp,
        'Train True negatives': best_tr_tn,
        'Train False positives': best_tr_fp,
        'Train False negatives': best_tr_fn,
        'Train Accuracy': best_tr_acc,
        'Train Precision': best_tr_pre,
        'Train Recall': best_tr_rec,
        'Train False positive rate': best_tr_fpr,
        'Train F-score': best_tr_fsc,
        'Test Evaluation time': best_tst_ev_tm,
        'Test True positives': best_tst_tp,
        'Test True negatives': best_tst_tn,
        'Test False positives': best_tst_fp,
        'Test False negatives': best_tst_fn,
        'Test Accuracy': best_tst_acc,
        'Test Precision': best_tst_pre,
        'Test Recall': best_tst_rec,
        'Test False positive rate': best_tst_fpr,
        'Test F-score': best_tst_fsc,
    })

    with pd.ExcelWriter(f'../Excel_reports/{args.xls_name}.xlsx', engine='openpyxl') as writer:

        # Write each dataframe to a different worksheet
        df1.to_excel(writer, sheet_name='Full')
        df2.to_excel(writer, sheet_name='Best')


if __name__ == "__main__":
    main()

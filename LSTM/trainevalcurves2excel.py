import os
import argparse

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # Create folder for report
    Path("../Analysis/Excel_reports").mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--xls_name', default='train_xls', help='Name of excel file to export')
    parser.add_argument('--archives_folder', default='default', help='Name of excel file to export')
    parser.add_argument('--n_thresh', type=int, default=38, help='Number of thresholds evaluated')
    parser.add_argument('--best', type=int, default=10,  help='Number of best models to save report')
    args = parser.parse_args()

    # working directory
    train_wkdir = '../Analysis/logs/train/' + args.archives_folder
    eval_wkdir = '../Analysis/logs/eval/' + args.archives_folder

    # Variable preallocating
    models = []
    params = []
    thresholds = []

    tr_time = []

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

    pr_auc = []
    roc_auc = []

    # Obtener los archivos de la carpeta
    train_files = os.listdir(train_wkdir)
    eval_files = os.listdir(eval_wkdir)

    # Leer tiempo de entrenamiento
    for fname in train_files:
        with open(os.path.join(train_wkdir, fname), 'r') as f:
            f.readline()
            f.readline()
            f.readline()

            params.extend([f.readline().split(":")[-1].strip()] * args.n_thresh)
            tr_time.extend([f.readline().split(":")[-1].strip()] * args.n_thresh)

    # Leer los archivos en la carpeta
    for fname in eval_files:
        with open(os.path.join(eval_wkdir, fname), 'r') as f:
            model_name = fname.split('.')[0]

            # Skip initial empty lines
            f.readline()
            f.readline()

            # Start reading threshold data
            for _ in range(args.n_thresh):
                models.append(model_name)

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
            # print(f'best thresh = {f.readline().split(":")[-2].split(",")[0].strip()}')
            # print(f'best fscore = {f.readline().split(":")[-1].strip()}')
            f.readline()
            f.readline()

            pr_auc.extend([f.readline().split(":")[-1].strip()] * args.n_thresh)
            roc_auc.extend([f.readline().split(":")[-1].strip()] * args.n_thresh)

    # Arrays to floats

    tp = list(map(float, tp))
    tn = list(map(float, tn))
    fp = list(map(float, fp))
    fn = list(map(float, fn))
    acc = list(map(float, acc))
    pre = list(map(float, pre))
    rec = list(map(float, rec))
    fpr = list(map(float, fpr))
    fsc = list(map(float, fsc))

    # params = list(map(float, params))

    pr_auc = list(map(float, pr_auc))
    roc_auc = list(map(float, roc_auc))

    # Get the 10 highest F-score models
    best_idx = np.argsort(fsc)

    best_models = [models[i] for i in best_idx[::-1][:args.best]]
    best_params = [params[i] for i in best_idx[::-1][:args.best]]
    best_tr_time = [tr_time[i] for i in best_idx[::-1][:args.best]]
    best_thresholds = [thresholds[i] for i in best_idx[::-1][:args.best]]
    best_ev_tm = [ev_tm[i] for i in best_idx[::-1][:args.best]]
    best_tp = [tp[i] for i in best_idx[::-1][:args.best]]
    best_tn = [tn[i] for i in best_idx[::-1][:args.best]]
    best_fp = [fp[i] for i in best_idx[::-1][:args.best]]
    best_fn = [fn[i] for i in best_idx[::-1][:args.best]]
    best_acc = [acc[i] for i in best_idx[::-1][:args.best]]
    best_pre = [pre[i] for i in best_idx[::-1][:args.best]]
    best_rec = [rec[i] for i in best_idx[::-1][:args.best]]
    best_fpr = [fpr[i] for i in best_idx[::-1][:args.best]]
    best_fsc = [fsc[i] for i in best_idx[::-1][:args.best]]
    best_pr_auc = [pr_auc[i] for i in best_idx[::-1][:args.best]]
    best_roc_auc = [roc_auc[i] for i in best_idx[::-1][:args.best]]

    df1 = pd.DataFrame({
        'Model_name': models,
        'Parameters': params,
        'Training time': tr_time,
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
        'PR AUC': pr_auc,
        'ROC AUC': roc_auc,
    })

    df2 = pd.DataFrame({
        'Model_name': best_models,
        'Parameters': best_params,
        'Training time': best_tr_time,
        'Threshold': best_thresholds,
        'Evaluation time': best_ev_tm,
        'True positives': best_tp,
        'True negatives': best_tn,
        'False positives': best_fp,
        'False negatives': best_fn,
        'Accuracy': best_acc,
        'Precision': best_pre,
        'Recall': best_rec,
        'False positive rate': best_fpr,
        'F-score': best_fsc,
        'PR AUC': best_pr_auc,
        'ROC AUC': best_roc_auc,
    })

    # Write report to excel
    with pd.ExcelWriter(f'../Analysis/Excel_reports/{args.xls_name}.xlsx', engine='openpyxl') as writer:

        # Write each dataframe to a different worksheet
        df1.to_excel(writer, sheet_name='Full', index=False)
        df2.to_excel(writer, sheet_name='Best', index=False)


if __name__ == "__main__":
    main()

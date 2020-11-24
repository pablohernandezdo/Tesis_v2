import os
import argparse

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # Create folder for report
    Path("../Analysis/Excel_reports").mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--xls_name', default='eval_xls', help='Name of excel file to export')
    parser.add_argument('--archives_folder', default='default', help='Name of excel file to export')
    parser.add_argument('--n_thresh', type=int, default=18, help='Number of thresholds evaluated')
    parser.add_argument('--best', type=int, default=10, help='Number of best models to save report')
    args = parser.parse_args()

    # working directory
    wkdir = '../Analysis/logs/' + args.archives_folder

    # Variable preallocating
    models = []
    thresholds = []

    francia_tp = []
    nevada_tp = []
    belgica_tp = []
    reykjanes_tp = []

    california_tn = []
    tides_tn = []
    utah_tn = []
    shaker_tn = []
    signals_tn = []

    francia_fn = []
    nevada_fn = []
    belgica_fn = []
    reykjanes_fn = []

    california_fp = []
    tides_fp = []
    utah_fp = []
    shaker_fp = []
    signals_fp = []

    tp = []
    tn = []
    fp = []
    fn = []

    acc = []
    pre = []
    rec = []
    fpr = []
    fsc = []

    pr_auc = []
    roc_auc = []

    # Obtener los archivos de la carpeta
    files = os.listdir(wkdir)

    # Leer tiempo de entrenamiento
    for fname in files:
        with open(os.path.join(wkdir, fname), 'r') as f:

            model_name = fname.split('.')[0]

            # Start reading threshold data
            for _ in range(args.n_thresh):
                models.append(model_name)

                thresh = f.readline().split(':')[-1].strip()
                thresholds.append(thresh)

                f.readline()

                francia_tp.append(f.readline().split(":")[-1].strip().split("/")[0])
                nevada_tp.append(f.readline().split(":")[-1].strip().split("/")[0])
                belgica_tp.append(f.readline().split(":")[-1].strip().split("/")[0])
                reykjanes_tp.append(f.readline().split(":")[-1].strip().split("/")[0])

                francia_fn.append(66-int(francia_tp[-1]))
                nevada_fn.append(8400-int(nevada_tp[-1]))
                belgica_fn.append(4-int(belgica_tp[-1]))
                reykjanes_fn.append(2552-int(reykjanes_tp[-1]))

                f.readline()

                california_tn.append(f.readline().split(":")[-1].strip().split("/")[0])
                tides_tn.append(f.readline().split(":")[-1].strip().split("/")[0])
                utah_tn.append(f.readline().split(":")[-1].strip().split("/")[0])
                shaker_tn.append(f.readline().split(":")[-1].strip().split("/")[0])
                signals_tn.append(f.readline().split(":")[-1].strip().split("/")[0])

                california_fp.append(834-int(california_tn[-1]))
                tides_fp.append(4318-int(tides_tn[-1]))
                utah_fp.append(1088-int(utah_tn[-1]))
                shaker_fp.append(1984-int(shaker_tn[-1]))
                signals_fp.append(699-int(signals_tn[-1]))

                f.readline()
                f.readline()
                f.readline()
                f.readline()

                tp.append(f.readline().split(":")[-1].strip())
                tn.append(f.readline().split(":")[-1].strip())
                fp.append(f.readline().split(":")[-1].strip())
                fn.append(f.readline().split(":")[-1].strip())

                f.readline()

                acc.append(f.readline().split(":")[-1].strip())
                pre.append(f.readline().split(":")[-1].strip())
                rec.append(f.readline().split(":")[-1].strip())
                fpr.append(f.readline().split(":")[-1].strip())
                fsc.append(f.readline().split(":")[-1].strip())

                f.readline()

            # best_thresh.append(f.readline().split(",")[0].split(":")[-1].strip())
            # best_fsc.append(f.readline().split(",")[1].split(":")[-1].strip())

            f.readline()

            pr_auc.extend([f.readline().split(":")[-1].strip()] * args.n_thresh)
            roc_auc.extend([f.readline().split(":")[-1].strip()] * args.n_thresh)
            # params.extend([f.readline().split(":")[-1].strip()] * args.n_thresh)

    francia_tp = list(map(float, francia_tp))
    nevada_tp = list(map(float, nevada_tp))
    belgica_tp = list(map(float, belgica_tp))
    reykjanes_tp = list(map(float, reykjanes_tp))

    california_tn = list(map(float, california_tn))
    tides_tn = list(map(float, tides_tn))
    utah_tn = list(map(float, utah_tn))
    shaker_tn = list(map(float, shaker_tn))
    signals_tn = list(map(float, signals_tn))

    francia_fn = list(map(float, francia_fn))
    nevada_fn = list(map(float, nevada_fn))
    belgica_fn = list(map(float, belgica_fn))
    reykjanes_fn = list(map(float, reykjanes_fn))

    california_fp = list(map(float, california_fp))
    tides_fp = list(map(float, tides_fp))
    utah_fp = list(map(float, utah_fp))
    shaker_fp = list(map(float, shaker_fp))
    signals_fp = list(map(float, signals_fp))

    tp = list(map(float, tp))
    tn = list(map(float, tn))
    fp = list(map(float, fp))
    fn = list(map(float, fn))

    acc = list(map(float, acc))
    pre = list(map(float, pre))
    rec = list(map(float, rec))
    fpr = list(map(float, fpr))
    fsc = list(map(float, fsc))

    thresholds = list(map(float, thresholds))

    pr_auc = list(map(float, pr_auc))
    roc_auc = list(map(float, roc_auc))

    # Get the 10 highest F-score models
    best_idx = np.argsort(fsc)

    best_models = [models[i] for i in best_idx[::-1][:args.best]]
    best_thresholds = [thresholds[i] for i in best_idx[::-1][:args.best]]
    best_francia_tp = [francia_tp[i] for i in best_idx[::-1][:args.best]]
    best_nevada_tp = [nevada_tp[i] for i in best_idx[::-1][:args.best]]
    best_belgica_tp = [belgica_tp[i] for i in best_idx[::-1][:args.best]]
    best_reykjanes_tp = [reykjanes_tp[i] for i in best_idx[::-1][:args.best]]
    best_california_tn = [california_tn[i] for i in best_idx[::-1][:args.best]]
    best_tides_tn = [tides_tn[i] for i in best_idx[::-1][:args.best]]
    best_utah_tn = [utah_tn[i] for i in best_idx[::-1][:args.best]]
    best_shaker_tn = [shaker_tn[i] for i in best_idx[::-1][:args.best]]
    best_signals_tn = [signals_tn[i] for i in best_idx[::-1][:args.best]]
    best_francia_fn = [francia_fn[i] for i in best_idx[::-1][:args.best]]
    best_nevada_fn = [nevada_fn[i] for i in best_idx[::-1][:args.best]]
    best_belgica_fn = [belgica_fn[i] for i in best_idx[::-1][:args.best]]
    best_reykjanes_fn = [reykjanes_fn[i] for i in best_idx[::-1][:args.best]]
    best_california_fp = [california_fp[i] for i in best_idx[::-1][:args.best]]
    best_tides_fp = [tides_fp[i] for i in best_idx[::-1][:args.best]]
    best_utah_fp = [utah_fp[i] for i in best_idx[::-1][:args.best]]
    best_shaker_fp = [shaker_fp[i] for i in best_idx[::-1][:args.best]]
    best_signals_fp = [signals_fp[i] for i in best_idx[::-1][:args.best]]
    best_tp = [fp[i] for i in best_idx[::-1][:args.best]]
    best_tn = [fn[i] for i in best_idx[::-1][:args.best]]
    best_fp = [fn[i] for i in best_idx[::-1][:args.best]]
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
        'Threshold': thresholds,
        'Francia TP:': francia_tp,
        'Francia FN:': francia_fn,
        'Nevada TP:': nevada_tp,
        'Nevada FN:': nevada_fn,
        'Belgica TP:': belgica_tp,
        'Belgica FN:': belgica_fn,
        'Reykjanes TP:': reykjanes_tp,
        'Reykjanes FN:': reykjanes_fn,
        'California TN:': california_tn,
        'California FP:': california_fp,
        'Tides TN:': tides_tn,
        'Tides FP:': tides_fp,
        'Utah TN:': utah_tn,
        'Utah FP:': utah_fp,
        'Shaker TN:': shaker_tn,
        'Shaker FP:': shaker_fp,
        'Signals TN:': signals_tn,
        'Signals FP:': signals_fp,
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
        'Threshold': best_thresholds,
        'Francia TP:': best_francia_tp,
        'Francia FN:': best_francia_fn,
        'Nevada TP:': best_nevada_tp,
        'Nevada FN:': best_nevada_fn,
        'Belgica TP:': best_belgica_tp,
        'Belgica FN:': best_belgica_fn,
        'Reykjanes TP:': best_reykjanes_tp,
        'Reykjanes FN:': best_reykjanes_fn,
        'California TN:': best_california_tn,
        'California FP:': best_california_fp,
        'Tides TN:': best_tides_tn,
        'Tides FP:': best_tides_fp,
        'Utah TN:': best_utah_tn,
        'Utah FP:': best_utah_fp,
        'Shaker TN:': best_shaker_tn,
        'Shaker FP:': best_shaker_fp,
        'Signals TN:': best_signals_tn,
        'Signals FP:': best_signals_fp,
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

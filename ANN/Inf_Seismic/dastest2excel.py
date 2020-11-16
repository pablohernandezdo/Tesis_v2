import os
import argparse

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # Create folder for report
    Path("../Excel_reports").mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--xls_name', default='eval_xls', help='Name of excel file to export')
    parser.add_argument('--archives_folder', default='default', help='Name of excel file to export')
    parser.add_argument('--n_thresh', type=int, default=18, help='Number of thresholds evaluated')
    args = parser.parse_args()

    # working directory
    wkdir = '../logs/' + args.archives_folder

    # Variable preallocating
    models = []
    params = []
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

                f.readline()

                california_tn.append(f.readline().split(":")[-1].strip().split("/")[0])
                tides_tn.append(f.readline().split(":")[-1].strip().split("/")[0])
                utah_tn.append(f.readline().split(":")[-1].strip().split("/")[0])
                shaker_tn.append(f.readline().split(":")[-1].strip().split("/")[0])
                signals_tn.append(f.readline().split(":")[-1].strip().split("/")[0])

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
            #params.extend([f.readline().split(":")[-1].strip()] * args.n_thresh)

    df = pd.DataFrame({
        'Model_name': models,
        #'Parameters': params,
        'Francia tp:': francia_tp,
        'Nevada tp:': nevada_tp,
        'Belgica tp:': belgica_tp,
        'Reykjanes tp:': reykjanes_tp,
        'California tn:': california_tn,
        'Tides tn:': tides_tn,
        'Utah tn:': utah_tn,
        'Shaker tn:': shaker_tn,
        'Signals tn:': signals_tn,
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

    df.to_excel(f'../Excel_reports/{args.xls_name}.xlsx', index=False)


if __name__ == "__main__":
    main()

import os
import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', default='eval_curves', help='Name of folder to read log files')
    parser.add_argument('--xls_name', default='train_xls', help='Name of excel file to export')
    parser.add_argument('--n_thresh', type=int, default=19, help='Number of thresholds evaluated')
    args = parser.parse_args()

    # working directory
    wkdir = os.path.join('logs', args.folder_name)

    # Variable preallocating
    models = []
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
            print(f'best thresh = {f.readline().split(":")[-2].split(",")[0].strip()}')
            print(f'best fscore = {f.readline().split(":")[-1].strip()}')

            # Skip empty line
            f.readline()

            print(f'PR AUC = {f.readline().split(":")[-1].strip()}')

            # Skip empty line, aqui hay que arreglar los reportes, deberia ser 1 readline
            f.readline()
            f.readline()

            print(f'ROC AUC = {f.readline().split(":")[-1].strip()}')


    df = pd.DataFrame({
        'Model_name': models,
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

    df.to_excel('test.xlsx', index=False)
    # Por cada archivo leer las lineas y extraer
    # la informacion importante, añadirla a un dataframe
    # Guardar el excel

    # df = pd.DataFrame({
    #     'Col A': [1, 2, 3],
    #     'Col B': ['A', 'B', 'C'],
    #     'Col C': ['Hola', 'Compas', 'Compitas'],
    # })
    #
    # df.to_excel('df.xlsx', index=False)


if __name__ == "__main__":
    main()

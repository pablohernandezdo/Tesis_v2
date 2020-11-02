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

    # Obtener los archivos de la carpeta
    files = os.listdir(wkdir)

    # Leer los archivos en la carpeta
    for fname in files:
        with open(os.path.join(wkdir, fname), 'r') as f:
            model_name = fname.split('.')[0]
            print(model_name)

            # Skip initial empty lines
            f.readline()
            f.readline()

            # Start reading threshold data
            for _ in range(args.n_thresh):
                thresh = f.readline().split(':')[-1].strip()
                thresholds.append(thresh)

                # Skip non-useful lines
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                # Read 4 cases
                print(f'tp = {f.readline().split(":")[-1].strip()}')
                print(f'tn = {f.readline().split(":")[-1].strip()}')
                print(f'fp = {f.readline().split(":")[-1].strip()}')
                print(f'fn = {f.readline().split(":")[-1].strip()}')

                # Skip empty line
                f.readline()

                # Read metrics
                print(f'acc = {f.readline().split(":")[-1].strip()}')
                print(f'pre = {f.readline().split(":")[-1].strip()}')
                print(f'rec = {f.readline().split(":")[-1].strip()}')
                print(f'fpr = {f.readline().split(":")[-1].strip()}')
                print(f'fscore = {f.readline().split(":")[-1].strip()}')

                # Skip empty line
                f.readline()

                print(f'eval_time = {f.readline().split(":")[-1].strip()}')

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

            print(thresholds)

        # Pa leer un solo archivo
        break

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

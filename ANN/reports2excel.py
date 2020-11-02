import os
import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', default='eval_curves', help='Name of folder to read log files')
    parser.add_argument('--xls_name', default='train_xls', help='Name of excel file to export')
    parser.add_argument('--n_thresh', type=int, default=10, help='Number of thresholds evaluated')
    args = parser.parse_args()

    # Obtener los archivos de la carpeta
    files = os.listdir(os.path.join('logs', args.folder_name))

    # Leer los archivos en la carpeta
    for fname in files:
        with open(fname, 'r') as f:
            print(f.readline())

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

import numpy as np
import scipy.io as sio


def main():
    f = sio.loadmat(
        "mat_2018_08_19_00h28m05s_Parkwind_HDAS_2Dmap_StrainData_2D.mat")
    traces = f['Data_2D']

    fs = 10
    n = traces.shape[1]
    t_ax = np.arange(n) / fs

    qbps = []

    for idx, tr in enumerate(traces):
        autocorr = np.correlate(tr, tr, mode='full')
        qbp = n * np.sum(np.power(autocorr, 2))
        qbps.append(qbp)
        print(idx)

    sorted_idxs = np.argsort(qbps)

    # sorted_qbps = [qbps[i] for i in sorted_idxs[::-1]]

    print(sorted_idxs[:20])


if __name__ == '__main__':
    main()

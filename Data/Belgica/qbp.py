import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy import signal


def main():
    f = sio.loadmat(
        "mat_2018_08_19_00h28m05s_Parkwind_HDAS_2Dmap_StrainData_2D.mat")
    traces = f['Data_2D']

    fs = 10
    n = traces.shape[1]
    t_ax = np.arange(n) / fs

    # avg_trace = np.mean(traces[3500:4001, :], 0)
    # avg_trace = signal.resample(avg_trace, n * 10)
    #
    # plt.plot(traces[126])
    # plt.show()

    qbps = []

    for idx, tr in enumerate(traces):
        autocorr = np.correlate(tr, tr, mode='full')
        qbp = n * np.sum(np.power(autocorr, 2))
        qbps.append(qbp)

    sorted_idxs = np.argsort(qbps)
    worst = sorted_idxs[-1429:]
    best = sorted_idxs[:10]

    # sorted_qbps = [qbps[i] for i in sorted_idxs[::-1]]

    print(qbps[sorted_idxs[0]] > qbps[sorted_idxs[0]])
    print('Mas ruidosas')
    for i in worst:
        print(i)

    print('Menos ruidosas')
    print(best)


if __name__ == '__main__':
    main()

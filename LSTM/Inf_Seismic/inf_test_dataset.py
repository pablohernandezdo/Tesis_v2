import re
import argparse
import itertools
from pathlib import Path

import h5py
import pywt
import torch
import segyio
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import default_rng
from scipy.signal import butter, lfilter

from model import *


def main():
    # Create images and animations folder
    Path("../Confusion_matrices").mkdir(exist_ok=True)
    Path("../PR_curves").mkdir(exist_ok=True)
    Path("../ROC_curves").mkdir(exist_ok=True)

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='CBN_1epch', help="Classifier model path")
    parser.add_argument("--classifier", default='CBN', help="Choose classifier architecture, C, CBN")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Load specified Classifier
    # if args.classifier == 'CBN':
    #     net = ClassConvBN()
    # elif args.classifier == 'CBN_v2':
    #     net = CBN_v2()
    # elif args.classifier == 'C':
    #     net = ClassConv()
    # else:
    #     net = ClassConv()
    #     print('Bad Classifier option, running classifier C')
    net = CNNLSTMANN()
    net.to(device)

    # Load parameters from trained model
    net.load_state_dict(torch.load('../../models/' + args.model_name + '.pth'))
    net.eval()

    # Preallocate precision and recall values
    precision = []
    fp_rate = []
    recall = []
    cm = []

    # Record max fscore value obtained
    max_fscore = 0

    # Record threshold of best fscore
    best_thresh = 0

    # Threshold values
    # thresholds = np.arange(0.4, 1, 0.05)
    thresholds = np.arange(0.1, 1, 0.1)
    # thresholds = np.linspace(0.05, 0.9, 18)
    # thresholds = np.linspace(0, 1, 11)
    # thresholds = np.linspace(0.4, 0.8, 5)
    # thresholds = np.linspace(0.5, 0.7, 11)

    # Round threshold values
    thresholds = np.around(thresholds, decimals=2)

    # For different threshold values
    for thresh in thresholds:
        print(f'THRESHOLD: {thresh}')

        # Count traces
        total_seismic, total_nseismic = 0, 0
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

        # Sismic classification
        total, tp, fn = inf_francia(net, device, thresh)
        total_seismic, total_tp, total_fn = sum_triple(total_seismic, total_tp, total_fn, total, tp, fn)

        total, tp, fn = inf_nevada(net, device, thresh)
        total_seismic, total_tp, total_fn = sum_triple(total_seismic, total_tp, total_fn, total, tp, fn)

        total, tp, fn = inf_belgica(net, device, thresh)
        total_seismic, total_tp, total_fn = sum_triple(total_seismic, total_tp, total_fn, total, tp, fn)

        total, tp, fn = inf_reykjanes(net, device, thresh)
        total_seismic, total_tp, total_fn = sum_triple(total_seismic, total_tp, total_fn, total, tp, fn)

        # # Non seismic classification
        total, tn, fp = inf_california(net, device, thresh)
        total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        # total, tn, fp = inf_hydraulic(net, device, thresh)
        # total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        total, tn, fp = inf_tides(net, device, thresh)
        total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        total, tn, fp = inf_utah(net, device, thresh)
        total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        total, tn, fp = inf_shaker(net, device, thresh)
        total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        total, tn, fp = inf_signals(net, device, thresh)
        total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        # Metrics
        pre, rec, fpr, fscore = print_metrics(total_seismic, total_nseismic, total_tp, total_fp, total_tn, total_fn)
        recall.append(rec)
        fp_rate.append(fpr)
        precision.append(pre)

        # Save best conf matrix
        if fscore > max_fscore:
            max_fscore = fscore
            cm = np.asarray([[total_tp, total_fn], [total_fp, total_tn]])
            best_thresh = thresh

    # PLOT BEST CONFUSION MATRIX
    target_names = ['Seismic', 'Non Seismic']

    # Confusion matrix
    plot_confusion_matrix(cm, target_names, title=f'Confusion matrix {args.model_name}, threshold = {best_thresh}',
                          filename=f'../Confusion_matrices/Confusion_matrix_{args.model_name}.png', normalize=False)

    # Normalized confusion matrix
    plot_confusion_matrix(cm, target_names, title=f'Confusion matrix {args.model_name}, threshold = {best_thresh}',
                          filename=f'../Confusion_matrices/Confusion_matrix_norm_{args.model_name}.png')

    # Precision/Recall curve
    plt.figure()
    plt.plot(recall, precision)

    # Annotate threshold values
    for i, j, k in zip(recall, precision, thresholds):
        plt.annotate(str(k), (i, j))

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'PR curve for model {args.model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'../PR_curves/PR_{args.model_name}.png')

    # Receiver operating characteristic curve
    plt.figure()
    plt.plot(fp_rate, recall)

    # Annotate
    for i, j, k in zip(fp_rate, recall, thresholds):
        plt.annotate(str(k), (i, j))

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curve for model {args.model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'../ROC_curves/ROC_{args.model_name}.png')


def normalize_trace(trace):
    trace = trace - np.mean(trace)
    trace = trace / np.max(np.abs(trace))
    return trace


def inf_francia(net, device, thresh):
    # Counters
    total, tp, fn = 0, 0, 0

    # Load Francia dataset
    f = scipy.io.loadmat("../../../Data/Francia/Earthquake_1p9_Var_BP_2p5_15Hz.mat")

    # Read data
    data = f["StrainFilt"]

    # Select some traces
    traces = []

    for trace in data:
        trace = trace - np.mean(trace)
        st = np.std(trace)

        if st > 50:
            traces.append(trace)

    traces = np.asarray(traces)
    traces = traces[:66]

    # For every trace in the file
    for trace in traces:
        if np.max(np.abs(trace)):
            # Normalize
            trace = trace / np.max(np.abs(trace))

            # Numpy to Torch
            trace = torch.from_numpy(trace).to(device).unsqueeze(0)

            # Prediction
            out_trace = net(trace.float())
            pred_trace = (out_trace > thresh)

            # Count traces
            total += 1

            if pred_trace:
                tp += 1
            else:
                fn += 1

    # Results
    print(f'Total Francia traces: {total}\n'
          f'True positives: {tp}\n'
          f'False negatives: {fn}\n')

    return total, tp, fn


def inf_nevada(net, device, thresh):
    # Counters
    total, tp, fn = 0, 0, 0

    # # Load Nevada data file 721
    # f = '../../../Data/Nevada/PoroTomo_iDAS16043_160321073721.sgy'

    # # Read data
    # with segyio.open(f, ignore_geometry=True) as segy:
    #     segy.mmap()
    #
    #     # Traces
    #     traces = segyio.tools.collect(segy.trace[:])
    #
    # # FALSE NEGATIVES = 941
    # # WITH ZERO PADDING 0 FN ALWAYS ?????
    # # npad = 1500
    #
    # # For every trace in the file
    # for trace in traces:
    #     # Resample
    #     trace = signal.resample(trace, 6000)
    #     # trace = signal.resample(trace, 3000)
    #
    #     # Zero padd
    #     # trace = np.pad(trace, (npad, npad), mode='constant')
    #
    #     # Normalize
    #     trace = trace / np.max(np.abs(trace))
    #
    #     # Numpy to Torch
    #     trace = torch.from_numpy(trace).to(device).unsqueeze(0)
    #
    #     # Prediction
    #     out_trace = net(trace.float())
    #     pred_trace = (out_trace > thresh).float().item()
    #
    #     # Count traces
    #     total += 1
    #
    #     if pred_trace:
    #         tp += 1
    #     else:
    #         fn += 1

    # Load Nevada data file 751
    f = '../../../Data/Nevada/PoroTomo_iDAS16043_160321073751.sgy'

    # For every trace in the file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # Select dataset basico traces
    tr1 = traces[50:2800]
    tr2 = traces[2900:4700]
    tr3 = traces[4800:8650]
    traces = np.vstack((tr1, tr2, tr3))

    # npad = 1500

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Zero pad
        # trace = np.pad(trace, (npad, npad), mode='constant')

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            tp += 1
        else:
            fn += 1

    # # Load Nevada dataset file 747
    # f = '../../../Data/Nevada/PoroTomo_iDAS025_160321073747.sgy'
    #
    # # Read data
    # with segyio.open(f, ignore_geometry=True) as segy:
    #     segy.mmap()
    #
    #     # Traces
    #     traces = segyio.tools.collect(segy.trace[:])
    #
    # # For every trace in the file
    # for trace in traces:
    #     # Resample
    #     trace = signal.resample(trace, 6000)
    #
    #     # Normalize
    #     trace = trace / np.max(np.abs(trace))
    #
    #     # Numpy to Torch
    #     trace = torch.from_numpy(trace).to(device).unsqueeze(0)
    #
    #     # Prediction
    #     out_trace = net(trace.float())
    #     pred_trace = (out_trace > thresh).float().item()
    #
    #     # Count traces
    #     total += 1
    #
    #     if pred_trace:
    #         tp += 1
    #     else:
    #         fn += 1

    # # Load Nevada dataset file 717
    # f = '../../../Data/Nevada/PoroTomo_iDAS025_160321073717.sgy'
    #
    # # Read data
    # with segyio.open(f, ignore_geometry=True) as segy:
    #     segy.mmap()
    #
    #     # Traces
    #     traces = segyio.tools.collect(segy.trace[:])
    #
    # # For every trace in the file
    # for trace in traces:
    #     # Resample
    #     trace = signal.resample(trace, 6000)
    #
    #     # Normalize
    #     trace = trace / np.max(np.abs(trace))
    #
    #     # Numpy to Torch
    #     trace = torch.from_numpy(trace).to(device).unsqueeze(0)
    #
    #     # Prediction
    #     out_trace = net(trace.float())
    #     pred_trace = (out_trace > thresh).float().item()
    #
    #     # Count traces
    #     total += 1
    #
    #     if pred_trace:
    #         tp += 1
    #     else:
    #         fn += 1

    # Results
    print(f'Total Nevada traces: {total}\n'
          f'True positives: {tp}\n'
          f'False negatives: {fn}\n')

    return total, tp, fn


def inf_belgica(net, device, thresh):
    # Counters
    total, tp, fn = 0, 0, 0

    # # Load Belgica data
    f = scipy.io.loadmat("../../../Data/Belgica/mat_2018_08_19_00h28m05s_Parkwind_HDAS_2Dmap_StrainData_2D.mat")

    # Read data
    traces = f['Data_2D']

    fs = 10

    # # For every trace in the file
    # for trace in traces:
    #     # Resample
    #     trace = signal.resample(trace, 6000)
    #
    #     # Normalize
    #     trace = trace / np.max(np.abs(trace))
    #
    #     # Numpy to Torch
    #     trace = torch.from_numpy(trace).to(device).unsqueeze(0)
    #
    #     # Prediction
    #     out_trace = net(trace.float())
    #     pred_trace = (out_trace > thresh).float().item()
    #
    #     # Count traces
    #     total_seismic += 1
    #
    #     if pred_trace:
    #         tp += 1
    #     else:
    #         fn += 1

    # Predict average 5km of measurements
    avg_data = np.mean(traces[3500:4001, :], 0)

    avg_fil1 = butter_bandpass_filter(avg_data, 0.5, 1, fs, order=5)
    avg_fil2 = butter_bandpass_filter(avg_data, 0.2, 0.6, 10, order=5)
    avg_fil3 = butter_bandpass_filter(avg_data, 0.1, 0.3, 10, order=5)

    traces = np.vstack((avg_data, avg_fil1, avg_fil2, avg_fil3))

    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        output = net(trace.float())

        predicted = (output > thresh)

        total += 1

        if predicted:
            tp += 1
        else:
            fn += 1

    # Results
    print(f'Total Belgica traces: {total}\n'
          f'True positives: {tp}\n'
          f'False negatives: {fn}\n')

    return total, tp, fn


def inf_reykjanes(net, device, thresh):
    # Rng
    rng = default_rng()

    # Counters
    total, tp, fn = 0, 0, 0

    # Reykjanes telesismo fibra optica
    file_fo = '../../../Data/Reykjanes/Jousset_et_al_2018_003_Figure3_fo.ascii'

    # Dict for header and data
    data_fo = {
        'head': '',
        'strain': []
    }

    # Read fo file and save content to data_fo
    with open(file_fo, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data_fo['head'] = line.strip()
            else:
                val = line.strip()
                data_fo['strain'].append(float(val))

    # Resample
    data_fo['strain'] = signal.resample(data_fo['strain'], 6000)

    # Normalize
    data_fo['strain'] = data_fo['strain'] / np.max(np.abs(data_fo['strain']))

    # Numpy to Torch
    data_fo['strain'] = torch.from_numpy(data_fo['strain']).to(device).unsqueeze(0)

    # Prediction
    out = net(data_fo['strain'].float())
    predicted = (out > thresh)

    # Count traces
    total += 1

    if predicted:
        tp += 1
    else:
        fn += 1

    # # Registro de sismo local con DAS
    file = '../../../Data/Reykjanes/Jousset_et_al_2018_003_Figure5b.ascii'
    n_trazas = 2551

    data = {
        'head': '',
        'strain': np.empty((1, n_trazas))
    }

    with open(file, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                data['head'] = line.strip()

            else:
                row = np.asarray(list(map(float, re.sub(' +', ' ', line).strip().split(' '))))
                data['strain'] = np.concatenate((data['strain'], np.expand_dims(row, 0)))

    data['strain'] = data['strain'][1:]
    traces = data['strain'].transpose()

    # Number of input samples to model
    final_samples = 6000

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 700)

        # Random place to put signal in
        idx = rng.choice(final_samples - len(trace), size=1)

        # Number of samples to zero pad on the right side
        right_pad = final_samples - idx - len(trace)

        # Zero pad signal
        trace = np.pad(trace, (idx.item(), right_pad.item()), mode='constant')

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            tp += 1
        else:
            fn += 1

    # Results
    print(f'Total Reykjanes traces: {total}\n'
          f'True positives: {tp}\n'
          f'False negatives: {fn}\n')

    return total, tp, fn


def inf_california(net, device, thresh):
    # Counters
    total, tn, fp = 0, 0, 0

    # # Load California dataset file
    f = scipy.io.loadmat('../../../Data/California/FSE-06_480SecP_SingDec_StepTest (1).mat')

    # Read data
    traces = f['singdecmatrix']
    traces = traces.transpose()

    # For every trace in the file
    for tr in traces:
        # Resample
        tr = signal.resample(tr, 41228)
        tr = tr[:36000]
        tr = np.reshape(tr, (-1, 6000))

        for trace in tr:
            # Normalize
            trace = trace / np.max(np.abs(trace))

            # Numpy to Torch
            trace = torch.from_numpy(trace).to(device).unsqueeze(0)

            # Prediction
            out_trace = net(trace.float())
            # pred_trace = torch.round(out_trace.data).item()
            pred_trace = (out_trace > thresh)

            # Count traces
            total += 1

            if pred_trace:
                fp += 1
            else:
                tn += 1

    # Results
    print(f'Total California traces: {total}\n'
          f'True negatives: {tn}\n'
          f'False positives: {fp}\n')

    return total, tn, fp


def inf_hydraulic(net, device, thresh):
    # Counters
    total, tn, fp = 0, 0, 0

    # File name
    file = '../../../Data/Hydraulic/CSULB500Pa600secP_141210183813.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['data'][()]

    # For every trace in the file
    for tr in traces:
        # Resample
        tr = signal.resample(tr, 12000)

        # Reshape
        tr = np.reshape(tr, (-1, 6000))

        for trace in tr:
            # Normalize
            trace = trace / np.max(np.abs(trace))

            # Numpy to Torch
            trace = torch.from_numpy(trace).to(device).unsqueeze(0)

            # Prediction
            out_trace = net(trace.float())
            pred_trace = (out_trace > thresh)

            # Count traces
            total += 1

            if pred_trace:
                fp += 1
            else:
                tn += 1

    file = '../../../Data/Hydraulic/CSULB500Pa10secP_141210174309.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['data'][()]

    # For every trace in the file
    for tr in traces:
        # Resample
        tr = signal.resample(tr, 205623)

        # Discard extra samples
        tr = tr[:(6000 * 34)]

        # Reshape
        tr = np.reshape(tr, (-1, 6000))

        for trace in tr:
            # Normalize
            trace = trace / np.max(np.abs(trace))

            # Numpy to Torch
            trace = torch.from_numpy(trace).to(device).unsqueeze(0)

            # Prediction
            out_trace = net(trace.float())
            pred_trace = (out_trace > thresh)

            # Count traces
            total += 1

            if pred_trace:
                fp += 1
            else:
                tn += 1

    file = '../../../Data/Hydraulic/CSULB500Pa100secP_141210175257.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['data'][()]

    # For every trace in the file
    for tr in traces:
        # Resample
        tr = signal.resample(tr, 600272)

        # Discard extra samples
        tr = tr[:600000]

        # Reshape
        tr = np.reshape(tr, (-1, 6000))

        for trace in tr:
            # Normalize
            trace = trace / np.max(np.abs(trace))

            # Numpy to Torch
            trace = torch.from_numpy(trace).to(device).unsqueeze(0)

            # Prediction
            out_trace = net(trace.float())
            pred_trace = (out_trace > thresh)

            # Count traces
            total += 1

            if pred_trace:
                fp += 1
            else:
                tn += 1

    # Results
    print(f'Total Hydraulic traces: {total}\n'
          f'True negatives: {tn}\n'
          f'False positives: {fp}\n')

    return total, tn, fp


def inf_tides(net, device, thresh):
    # Counters
    total, tn, fp = 0, 0, 0

    # File name
    file = '../../../Data/Tides/CSULB_T13_EarthTide_earthtide_mean_360_519.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['clipdata'][()]

    # Resample to 100 Hz
    traces = signal.resample(traces, 25909416)

    # Discard extra samples
    traces = traces[:25908000]

    # Reshape to matrix of traces
    traces = traces.reshape(-1, 6000)

    # For every trace in the file
    for trace in traces:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Results
    print(f'Total Tides traces: {total}\n'
          f'True negatives: {tn}\n'
          f'False positives: {fp}\n')

    return total, tn, fp


def inf_utah(net, device, thresh):
    # Counters
    total, tn, fp = 0, 0, 0

    # Load Utah data file 1
    f = '../../../Data/Utah/FORGE_78-32_iDASv3-P11_UTC190419001218.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Results
    print(f'Total Utah traces: {total}\n'
          f'True negatives: {tn}\n'
          f'False positives: {fp}\n')

    return total, tn, fp


def inf_vibroseis(net, device, thresh):
    # Counters
    total, tn, fp = 0, 0, 0

    # Load Vibroseis dataset file 047
    f = '../../../Data/Vibroseis/PoroTomo_iDAS025_160325140047.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Load Nevada dataset file 117
    f = '../../../Data/Vibroseis/PoroTomo_iDAS025_160325140117.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Load Nevada dataset file 048
    f = '../../../Data/Vibroseis/PoroTomo_iDAS16043_160325140048.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Load Vibroseis data file 118
    f = '../../../Data/Vibroseis/PoroTomo_iDAS16043_160325140118.sgy'

    # For every trace in the file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6000)

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Results
    print(f'Total Vibroseis traces: {total}\n'
          f'True negatives: {tn}\n'
          f'False positives: {fp}\n')

    return total, tn, fp


def inf_shaker(net, device, thresh):
    # Counters
    total, tn, fp = 0, 0, 0

    # Load Shaker dataset file
    f = '../../../Data/Shaker/large shaker NEES_130910161319 (1).sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    for trace in traces:
        # Resample
        trace = signal.resample(trace, 6300)

        # Discard last 300 samples
        trace = trace[:6000]

        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Results
    print(f'Total Shaker traces: {total}\n'
          f'True negatives: {tn}\n'
          f'False positives: {fp}\n')

    return total, tn, fp


def inf_signals(net, device, thresh):
    # Counters
    total, tn, fp = 0, 0, 0

    # Señales de prueba
    # Init rng
    rng = default_rng()

    sig_prueba = 0

    # Noise
    ns = rng.normal(0, 1, 6000)

    sig_prueba += 1

    # Sine waves

    # Number of sample points
    N = 6000

    # sampling frequency
    fs = 100

    # Time axis
    t = np.linspace(0.0, N / fs, N)

    # Number of frequency interval steps
    n = 100

    # Frequency spans
    fr1 = np.linspace(1, 50, n)
    fr2 = np.linspace(0.01, 1, n)

    # Prealocate
    wvs1 = []
    wvs2 = []
    wvs3 = []

    for f1, f2 in zip(fr1, fr2):
        sig1 = np.sin(f1 * 2.0 * np.pi * t)
        sig2 = np.sin(f2 * 2.0 * np.pi * t)
        wvs1.append(sig1)
        wvs2.append(sig2)
        wvs3.append(sig1 + sig2)

    wvs1 = np.array(wvs1)
    wvs2 = np.array(wvs2)
    wvs3 = np.array(wvs3)

    wvs1_ns = wvs1 + 0.5 * rng.normal(0, 1, wvs1.shape)
    wvs2_ns = wvs2 + 0.5 * rng.normal(0, 1, wvs2.shape)
    wvs3_ns = wvs3 + 0.5 * rng.normal(0, 1, wvs3.shape)

    sig_prueba += len(wvs1)
    sig_prueba += len(wvs2)
    sig_prueba += len(wvs3)
    sig_prueba += len(wvs1_ns)
    sig_prueba += len(wvs2_ns)
    sig_prueba += len(wvs3_ns)

    # PADDED SINES

    # Number of intermediate sample points
    ni = [1000, 2000, 4000, 5000]

    # Number of points to zero-pad
    pad = [(N - n) // 2 for n in ni]

    # Time axis for smaller waves
    lts = [np.linspace(0.0, nis / fs, nis) for nis in ni]

    # All frequencies list
    all_fr = []

    # Calculate max period for smaller waves
    max_periods = [n_points / fs for n_points in ni]

    # Calculate frequencies for smaller waves
    for per in max_periods:
        freqs = []
        for i in range(1, int(per) + 1):
            if per % i == 0:
                freqs.append(1 / i)
        all_fr.append(freqs)

    # Preallocate waves
    wvs_pad = []

    # Generate waves and zero pad
    for idx, fr_ls in enumerate(all_fr):
        for fr in fr_ls:
            wv = np.sin(fr * 2.0 * np.pi * lts[idx])
            wv = np.pad(wv, (pad[idx], pad[idx]), 'constant')
            wvs_pad.append(wv)

    sig_prueba += len(wvs_pad)

    # Wavelets

    # Preallocate wavelets
    lets = []

    # Discrete wavelet families
    discrete_families = ['db', 'sym', 'coif', 'bior', 'rbio']

    # Obtain wavelet waveforms, resample and append
    for fam in discrete_families:
        for wavelet in pywt.wavelist(fam):
            wv = pywt.Wavelet(wavelet)
            if wv.orthogonal:
                [_, psi, _] = pywt.Wavelet(wavelet).wavefun(level=5)
                psi = signal.resample(psi, 6000)
                lets.append(psi)

    sig_prueba += len(lets)

    # Normalize
    ns_norm = ns / np.max(np.abs(ns))

    # Numpy to Torch
    ns_norm = torch.from_numpy(ns_norm).to(device).unsqueeze(0)

    # Prediction
    out = net(ns_norm.float())

    out = (out > thresh)

    total += 1

    if out:
        fp += 1
    else:
        tn += 1

    # For every trace in wvs1
    for trace in wvs1:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # For every trace in wvs1
    for trace in wvs2:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # For every trace in wvs1
    for trace in wvs3:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # For every trace in wvs1
    for trace in wvs1_ns:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    for trace in wvs2_ns:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    for trace in wvs3_ns:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    for trace in wvs_pad:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    for trace in lets:
        # Normalize
        trace = trace / np.max(np.abs(trace))

        # Numpy to Torch
        trace = torch.from_numpy(trace).to(device).unsqueeze(0)

        # Prediction
        out_trace = net(trace.float())
        pred_trace = (out_trace > thresh)

        # Count traces
        total += 1

        if pred_trace:
            fp += 1
        else:
            tn += 1

    # Results
    print(f'Total test signals: {total}\n'
          f'True negatives: {tn}\n'
          f'False positives: {fp}\n')

    return total, tn, fp


def sum_triple(i1, i2, i3, s1, s2, s3):
    return s1 + i1, s2 + i2, s3 + i3


def print_metrics(total_seismic, total_nseismic, tp, fp, tn, fn):

    acc = (tp + tn) / (tp + fp + tn +fn)

    # Evaluation metrics
    if (not tp) and (not fp):
        precision = 1
    else:
        precision = tp / (tp + fp)

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    if (not precision) and (not recall):
        fscore = 0
    else:
        fscore = 2 * (precision * recall) / (precision + recall)

    # Results
    print(f'Total seismic traces: {total_seismic}\n'
          f'Total non seismic traces: {total_nseismic}\n\n'
          f'True positives: {tp}\n'
          f'True negatives: {tn}\n'
          f'False positives: {fp}\n'
          f'False negatives: {fn}\n\n'
          f'Accuracy: {acc:5.3f}\n'
          f'Precision: {precision:5.3f}\n'
          f'Recall: {recall:5.3f}\n'
          f'F-score: {fscore:5.3f}\n'
          f'False positive rate: {fpr:5.3f}\n')

    return precision, recall, fpr, fscore


# ESTA FUNCION LA SAQUÉ DE https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
# HAY QUE MODIFICARLA PA QUE SEA MAS MEJOR
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', filename='Confusion_matrix.png', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(filename)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', output='ba')
    return b, a


def butter_bandpass_filter(dat, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, dat)
    return y


if __name__ == "__main__":
    main()

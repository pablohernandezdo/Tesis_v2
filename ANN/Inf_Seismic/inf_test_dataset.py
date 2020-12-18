import re
import argparse
import itertools
from pathlib import Path

import tqdm
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
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", default='default', help="Folder to save model")
    parser.add_argument("--model_name", default='CBN_1epch', help="Classifier model path")
    parser.add_argument("--classifier", default='C', help="Choose classifier architecture")
    args = parser.parse_args()

    # Create curves folders
    Path(f"../Analysis/Confusion_matrices/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/PR_curves/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/ROC_curves/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/Fscore_curves/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/FPFN_curves/{args.model_folder}").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/Histograms/{args.model_folder}").mkdir(parents=True, exist_ok=True)

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Load parameters from trained model
    net.load_state_dict(torch.load('../../models/' + args.model_folder + '/' + args.model_name + '.pth'))
    net.eval()

    # Seismic and non seismic output values
    hist = 1
    s_outputs = []
    ns_outputs = []

    # Preallocate precision and recall values
    precision = []
    fp_rate = []
    recall = []
    fscores = []

    fp_plt = []
    fn_plt = []

    # COnfusion matrix
    cm = []

    # Record max fscore value obtained
    max_fscore = 0

    # Record threshold of best fscore
    best_thresh = 0

    # Threshold values
    # thresholds = np.arange(0.05, 1, 0.05)
    thresholds = np.array([0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475])

    # Round threshold values
    thresholds = np.around(thresholds, decimals=3)

    # Evaluate model on DAS test dataset

    for thresh in thresholds:

        # Count traces
        total_seismic, total_nseismic = 0, 0
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

        # Print threshold value
        print(f'Threshold value: {thresh}\n')

        # Sismic classification
        total, tp, fn, outs = inf_francia(net, device, thresh, hist)
        s_outputs.extend(outs)
        total_seismic, total_tp, total_fn = sum_triple(total_seismic, total_tp, total_fn, total, tp, fn)

        total, tp, fn, outs = inf_nevada(net, device, thresh, hist)
        s_outputs.extend(outs)
        total_seismic, total_tp, total_fn = sum_triple(total_seismic, total_tp, total_fn, total, tp, fn)

        total, tp, fn, outs = inf_belgica(net, device, thresh, hist)
        s_outputs.extend(outs)
        total_seismic, total_tp, total_fn = sum_triple(total_seismic, total_tp, total_fn, total, tp, fn)

        total, tp, fn, outs = inf_reykjanes(net, device, thresh, hist)
        s_outputs.extend(outs)
        total_seismic, total_tp, total_fn = sum_triple(total_seismic, total_tp, total_fn, total, tp, fn)

        # # Non seismic classification
        total, tn, fp, outs = inf_california(net, device, thresh, hist)
        ns_outputs.extend(outs)
        total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        # total, tn, fp, outs = inf_hydraulic(net, device, thresh, hist)
        # ns_outputs.extend(outs)
        # total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        total, tn, fp, outs = inf_tides(net, device, thresh, hist)
        ns_outputs.extend(outs)
        total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        total, tn, fp, outs = inf_utah(net, device, thresh, hist)
        ns_outputs.extend(outs)
        total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        total, tn, fp, outs = inf_shaker(net, device, thresh, hist)
        ns_outputs.extend(outs)
        total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        total, tn, fp, outs = inf_signals(net, device, thresh, hist)
        ns_outputs.extend(outs)
        total_nseismic, total_tn, total_fp = sum_triple(total_nseismic, total_tn, total_fp, total, tn, fp)

        # Run hist just once
        hist = 0

        # Metrics
        pre, rec, fpr, fscore = print_metrics(total_seismic, total_nseismic, total_tp, total_fp, total_tn, total_fn)
        recall.append(rec)
        fp_rate.append(fpr)
        precision.append(pre)
        fscores.append(fscore)

        fp_plt.append(fp)
        fn_plt.append(fn)

        # Save best conf matrix
        if fscore > max_fscore:
            max_fscore = fscore
            cm = np.asarray([[total_tp, total_fn], [total_fp, total_tn]])
            best_thresh = thresh

    # Add point (0, 1) to PR curve
    precision.append(1)
    recall.append(0)

    # Add point (1, 0.5) to PR curve
    precision.insert(0, 0.5)
    recall.insert(0, 1)

    # Add point (0, 0)  to ROC curve
    fp_rate.append(0)

    # Add point (1, 1) to ROC curve
    fp_rate.insert(0, 1)

    # Area under curves
    pr_auc = np.trapz(precision[::-1], x=recall[::-1])
    roc_auc = np.trapz(recall[::-1], x=fp_rate[::-1])

    # Print fscores
    print(f'Best threshold: {best_thresh}, f-score: {max_fscore:5.3f}\n'
          f'PR AUC: {pr_auc:5.3f}\n'
          f'ROC AUC: {roc_auc:5.3f}\n')

    # Plot histograms
    plot_histograms(s_outputs, ns_outputs, args.model_folder, args.model_name)

    # PLOT BEST CONFUSION MATRIX
    target_names = ['Seismic', 'Non Seismic']

    # Normalized confusion matrix
    plot_confusion_matrix(cm, target_names,
                          title=f'Confusion matrix {args.model_name}, threshold = {best_thresh}',
                          filename=f'../Analysis/Confusion_matrices/{args.model_folder}/Confusion_matrix_das_{args.model_name}.png')

    # F-score vs thresholds curve
    plt.figure()
    plt.plot(thresholds, fscores)
    plt.title(f'Fscores por umbral modelo {args.model_name}')
    plt.xlabel('Umbrales')
    plt.ylabel('F-score')
    plt.grid(True)
    plt.savefig(f'../Analysis/Fscore_curves/{args.model_folder}/Fscore_{args.model_name}.png')

    # False positives / False negatives curve
    plt.figure()
    line_fp, = plt.plot(thresholds, fp_plt, label='False positives')
    line_fn, = plt.plot(thresholds, fn_plt, label='False negatives')

    plt.title(f'FP y FN modelo {args.model_name}')
    plt.xlabel('Umbrales')
    plt.ylabel('Total')
    plt.grid(True)
    plt.legend(handles=[line_fp, line_fn], loc='best')
    plt.savefig(f'../Analysis/FPFN_curves/{args.model_folder}/FPFN_{args.model_name}.png')

    # Precision/Recall curve
    plt.figure()
    plt.plot(recall, precision, '-o', markersize=4)

    # Annotate threshold values
    for i, j, k in zip(recall, precision, thresholds):
        plt.annotate(str(k), (i, j))

    # Dumb model line
    plt.hlines(0.5, 0, 1, 'b', '--')
    plt.title(f'PR curve for model {args.model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(-0.02, 1.02)
    plt.ylim(0.48, 1.02)
    plt.grid(True)
    plt.savefig(f'../Analysis/PR_curves/{args.model_folder}/PR_{args.model_name}.png')

    # Receiver operating characteristic curve
    plt.figure()
    plt.plot(fp_rate, recall, '-o', markersize=4)

    # Annotate
    for i, j, k in zip(fp_rate, recall, thresholds):
        plt.annotate(str(k), (i, j))

    # Dumb model line
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(f'ROC curve for model {args.model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.savefig(f'../Analysis/ROC_curves/{args.model_folder}/ROC_{args.model_name}.png')


def normalize_trace(trace):
    trace = trace - np.mean(trace)
    trace = trace / np.max(np.abs(trace))
    return trace


def plot_histograms(s_outputs, ns_outputs, model_folder, model_name):

    plt.figure()

    n_seis, bins_seis, patches_seis = plt.hist(s_outputs, bins=100, color='blue', alpha=0.6, label='Seismic')
    n_nseis, bins_nseis, patches_nseis = plt.hist(ns_outputs, bins=100, color='red', alpha=0.6, label='Non seismic')

    plt.title(f'Output values histogram model {model_name}')
    plt.xlabel('Net output value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(f'../Analysis/Histograms/{model_folder}/Histogram_{model_name}.png')


def inf_francia(net, device, thresh, hist):
    # Counters
    total, tp, fn = 0, 0, 0

    # Outputs list
    outputs = []

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
    with tqdm.tqdm(total=len(traces), desc='Francia dataset evaluation') as francia_bar:
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

                if hist:
                    outputs.append(out_trace.item())

                if pred_trace:
                    tp += 1
                else:
                    fn += 1

                francia_bar.update()

    # Results
    print(f'Francia true positives: {tp}/{total}')

    return total, tp, fn, outputs


def inf_nevada(net, device, thresh, hist):
    # Counters
    total, tp, fn = 0, 0, 0

    # Outputs list
    outputs = []

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
    #     if hist:
    #       outputs.append(out_trace.item())
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
    with tqdm.tqdm(total=len(traces), desc='Nevada dataset evaluation') as nevada_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                tp += 1
            else:
                fn += 1

            nevada_bar.update()

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
    #     if hist:
    #       outputs.append(out_trace.item())
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
    #     if hist:
    #         outputs.append(out_trace.item())
    #
    #     if pred_trace:
    #         tp += 1
    #     else:
    #         fn += 1

    # Results
    print(f'Nevada true positives: {tp}/{total}')

    return total, tp, fn, outputs


def inf_belgica(net, device, thresh, hist):
    # Counters
    total, tp, fn = 0, 0, 0

    # Outputs list
    outputs = []

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
    #     if hist:
    #         outputs.append(out_trace.item())
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

    with tqdm.tqdm(total=len(traces), desc='Belgica dataset evaluation') as belgica_bar:
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

            if hist:
                outputs.append(output.item())

            if predicted:
                tp += 1
            else:
                fn += 1

            belgica_bar.update()

    # Results
    print(f'Belgica true positives: {tp}/{total}')

    return total, tp, fn, outputs


def inf_reykjanes(net, device, thresh, hist):
    # Rng
    rng = default_rng()

    # Counters
    total, tp, fn = 0, 0, 0

    # Outputs list
    outputs = []

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

    if hist:
        outputs.append(out.item())

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
    with tqdm.tqdm(total=len(traces), desc='Reykjanes dataset evaluation') as reykjanes_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                tp += 1
            else:
                fn += 1

            reykjanes_bar.update()

    # Results
    print(f'Reykjanes true positives: {tp}/{total}\n')

    return total, tp, fn, outputs


def inf_california(net, device, thresh, hist):
    # Counters
    total, tn, fp = 0, 0, 0

    # Outputs list
    outputs = []

    # # Load California dataset file
    f = scipy.io.loadmat('../../../Data/California/FSE-06_480SecP_SingDec_StepTest (1).mat')

    # Read data
    traces = f['singdecmatrix']
    traces = traces.transpose()

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='California dataset evaluation') as california_bar:
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

                if hist:
                    outputs.append(out_trace.item())

                if pred_trace:
                    fp += 1
                else:
                    tn += 1

                california_bar.update()

    # Results
    print(f'California true negatives: {tn}/{total}')

    return total, tn, fp, outputs


def inf_hydraulic(net, device, thresh, hist):
    # Counters
    total, tn, fp = 0, 0, 0

    # Outputs list
    outputs = []

    # File name
    file = '../../../Data/Hydraulic/CSULB500Pa600secP_141210183813.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['data'][()]

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='Hydraulic dataset 1 evaluation') as hydraulic1_bar:
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

                if hist:
                    outputs.append(out_trace.item())

                if pred_trace:
                    fp += 1
                else:
                    tn += 1

                hydraulic1_bar.update()

    file = '../../../Data/Hydraulic/CSULB500Pa10secP_141210174309.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['data'][()]

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='Hydraulic dataset 2 evaluation') as hydraulic2_bar:
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

                if hist:
                    outputs.append(out_trace.item())

                if pred_trace:
                    fp += 1
                else:
                    tn += 1

                hydraulic2_bar.update()

    file = '../../../Data/Hydraulic/CSULB500Pa100secP_141210175257.mat'

    # Read file data
    with h5py.File(file, 'r') as f:
        traces = f['data'][()]

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='Hydraulic dataset 3 evaluation') as hydraulic3_bar:
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

                if hist:
                    outputs.append(out_trace.item())

                if pred_trace:
                    fp += 1
                else:
                    tn += 1

                hydraulic3_bar.update()

    # Results
    print(f'Hydraulic true negatives: {tn}/{total}')

    return total, tn, fp, outputs


def inf_tides(net, device, thresh, hist):
    # Counters
    total, tn, fp = 0, 0, 0

    # Outputs list
    outputs = []

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
    with tqdm.tqdm(total=len(traces), desc='Tides dataset evaluation') as tides_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            tides_bar.update()

    # Results
    print(f'Tides true negatives: {tn}/{total}')

    return total, tn, fp, outputs


def inf_utah(net, device, thresh, hist):
    # Counters
    total, tn, fp = 0, 0, 0

    # Outputs list
    outputs = []

    # Load Utah data file 1
    f = '../../../Data/Utah/FORGE_78-32_iDASv3-P11_UTC190419001218.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='Utah dataset evaluation') as utah_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            utah_bar.update()

    # Results
    print(f'Utah true negatives: {tn}/{total}')

    return total, tn, fp, outputs


def inf_vibroseis(net, device, thresh, hist):
    # Counters
    total, tn, fp = 0, 0, 0

    # Outputs list
    outputs = []

    # Load Vibroseis dataset file 047
    f = '../../../Data/Vibroseis/PoroTomo_iDAS025_160325140047.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='Vibroseis dataset 047 evaluation') as vibroseis1_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            vibroseis1_bar.update()

    # Load Nevada dataset file 117
    f = '../../../Data/Vibroseis/PoroTomo_iDAS025_160325140117.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='Vibroseis dataset 117 evaluation') as vibroseis2_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            vibroseis2_bar.update()

    # Load Nevada dataset file 048
    f = '../../../Data/Vibroseis/PoroTomo_iDAS16043_160325140048.sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='Vibroseis dataset 048 evaluation') as vibroseis3_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            vibroseis3_bar.update()

    # Load Vibroseis data file 118
    f = '../../../Data/Vibroseis/PoroTomo_iDAS16043_160325140118.sgy'

    # For every trace in the file
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='Vibroseis dataset 118 evaluation') as vibroseis4_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            vibroseis4_bar.update()

    # Results
    print(f'Vibroseis true negatives: {tn}/{total}')

    return total, tn, fp, outputs


def inf_shaker(net, device, thresh, hist):
    # Counters
    total, tn, fp = 0, 0, 0

    # Outputs list
    outputs = []

    # Load Shaker dataset file
    f = '../../../Data/Shaker/large shaker NEES_130910161319 (1).sgy'

    # Read data
    with segyio.open(f, ignore_geometry=True) as segy:
        segy.mmap()

        # Traces
        traces = segyio.tools.collect(segy.trace[:])

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='Shaker dataset evaluation') as shaker_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            shaker_bar.update()

    # Results
    print(f'Shaker true negatives: {tn}/{total}')

    return total, tn, fp, outputs


def inf_signals(net, device, thresh, hist):
    # Counters
    total, tn, fp = 0, 0, 0

    # Outputs list
    outputs = []

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

    if hist:
        outputs.append(out.item())

    if out:
        fp += 1
    else:
        tn += 1

    # For every trace in wvs1
    with tqdm.tqdm(total=len(wvs1), desc='Signals wvs1 evaluation') as wvs1_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            wvs1_bar.update()

    # For every trace in wvs1
    with tqdm.tqdm(total=len(wvs2), desc='Signals wvs2 evaluation') as wvs2_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            wvs2_bar.update()

    # For every trace in wvs1
    with tqdm.tqdm(total=len(wvs3), desc='Signals wvs3 evaluation') as wvs3_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            wvs3_bar.update()

    # For every trace in wvs1
    with tqdm.tqdm(total=len(wvs1_ns), desc='Signals wvs1_ns evaluation') as wvs1_ns_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            wvs1_ns_bar.update()

    with tqdm.tqdm(total=len(wvs2_ns), desc='Signals wvs2_ns evaluation') as wvs2_ns_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            wvs2_ns_bar.update()

    with tqdm.tqdm(total=len(wvs3_ns), desc='Signals wvs3_ns evaluation') as wvs3_ns_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            wvs3_ns_bar.update()

    with tqdm.tqdm(total=len(wvs_pad), desc='Signals wvs_pad evaluation') as wvs_pad_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            wvs_pad_bar.update()

    with tqdm.tqdm(total=len(lets), desc='Signals wavelets evaluation') as wvlets_bar:
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

            if hist:
                outputs.append(out_trace.item())

            if pred_trace:
                fp += 1
            else:
                tn += 1

            wvlets_bar.update()

    # Results
    print(f'Test signals true negatives: {tn}/{total}\n')

    return total, tn, fp, outputs


def sum_triple(i1, i2, i3, s1, s2, s3):
    return s1 + i1, s2 + i2, s3 + i3


def print_metrics(total_seismic, total_nseismic, tp, fp, tn, fn):

    acc = (tp + tn) / (tp + fp + tn + fn)

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
          f'False positive rate: {fpr:5.3f}\n'
          f'F-score: {fscore:5.3f}\n')

    return precision, recall, fpr, fscore


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                          filename='Confusion_matrix.png', cmap=None, normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    missclass = 1 - accuracy

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
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, missclass))
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


def get_classifier(x):
    return {
        '1h6k': OneHidden6k(),
        '1h5k': OneHidden5k(),
        '1h4k': OneHidden4k(),
        '1h3k': OneHidden3k(),
        '1h2k': OneHidden2k(),
        '1h1k': OneHidden1k(),
        '1h5h': OneHidden5h(),
        '1h1h': OneHidden1h(),
        '1h10': OneHidden10(),
        '1h1': OneHidden1(),
        '2h6k6k': TwoHidden6k6k(),
        '2h6k5k': TwoHidden6k5k(),
        '2h6k4k': TwoHidden6k4k(),
        '2h6k3k': TwoHidden6k3k(),
        '2h6k2k': TwoHidden6k2k(),
        '2h6k1k': TwoHidden6k1k(),
        '2h6k5h': TwoHidden6k5h(),
        '2h6k1h': TwoHidden6k1h(),
        '2h6k10': TwoHidden6k10(),
        '2h6k1': TwoHidden6k1(),
        '2h5k6k': TwoHidden5k6k(),
        '2h5k5k': TwoHidden5k5k(),
        '2h5k4k': TwoHidden5k4k(),
        '2h5k3k': TwoHidden5k3k(),
        '2h5k2k': TwoHidden5k2k(),
        '2h5k1k': TwoHidden5k1k(),
        '2h5k5h': TwoHidden5k5h(),
        '2h5k1h': TwoHidden5k1h(),
        '2h5k10': TwoHidden5k10(),
        '2h5k1': TwoHidden5k1(),
        '2h4k6k': TwoHidden4k6k(),
        '2h4k5k': TwoHidden4k5k(),
        '2h4k4k': TwoHidden4k4k(),
        '2h4k3k': TwoHidden4k3k(),
        '2h4k2k': TwoHidden4k2k(),
        '2h4k1k': TwoHidden4k1k(),
        '2h4k5h': TwoHidden4k5h(),
        '2h4k1h': TwoHidden4k1h(),
        '2h4k10': TwoHidden4k10(),
        '2h4k1': TwoHidden4k1(),
        '2h3k6k': TwoHidden3k6k(),
        '2h3k5k': TwoHidden3k5k(),
        '2h3k4k': TwoHidden3k4k(),
        '2h3k3k': TwoHidden3k3k(),
        '2h3k2k': TwoHidden3k2k(),
        '2h3k1k': TwoHidden3k1k(),
        '2h3k5h': TwoHidden3k5h(),
        '2h3k1h': TwoHidden3k1h(),
        '2h3k10': TwoHidden3k10(),
        '2h3k1': TwoHidden3k1(),
        '2h2k6k': TwoHidden2k6k(),
        '2h2k5k': TwoHidden2k5k(),
        '2h2k4k': TwoHidden2k4k(),
        '2h2k3k': TwoHidden2k3k(),
        '2h2k2k': TwoHidden2k2k(),
        '2h2k1k': TwoHidden2k1k(),
        '2h2k5h': TwoHidden2k5h(),
        '2h2k1h': TwoHidden2k1h(),
        '2h2k10': TwoHidden2k10(),
        '2h2k1': TwoHidden2k1(),
        '2h1k6k': TwoHidden1k6k(),
        '2h1k5k': TwoHidden1k5k(),
        '2h1k4k': TwoHidden1k4k(),
        '2h1k3k': TwoHidden1k3k(),
        '2h1k2k': TwoHidden1k2k(),
        '2h1k1k': TwoHidden1k1k(),
        '2h1k5h': TwoHidden1k5h(),
        '2h1k1h': TwoHidden1k1h(),
        '2h1k10': TwoHidden1k10(),
        '2h1k1': TwoHidden1k1(),
        '2h5h6k': TwoHidden5h6k(),
        '2h5h5k': TwoHidden5h5k(),
        '2h5h4k': TwoHidden5h4k(),
        '2h5h3k': TwoHidden5h3k(),
        '2h5h2k': TwoHidden5h2k(),
        '2h5h1k': TwoHidden5h1k(),
        '2h5h5h': TwoHidden5h5h(),
        '2h5h1h': TwoHidden5h1h(),
        '2h5h10': TwoHidden5h10(),
        '2h5h1': TwoHidden5h1(),
        '2h1h6k': TwoHidden1h6k(),
        '2h1h5k': TwoHidden1h5k(),
        '2h1h4k': TwoHidden1h4k(),
        '2h1h3k': TwoHidden1h3k(),
        '2h1h2k': TwoHidden1h2k(),
        '2h1h1k': TwoHidden1h1k(),
        '2h1h5h': TwoHidden1h5h(),
        '2h1h1h': TwoHidden1h1h(),
        '2h1h10': TwoHidden1h10(),
        '2h1h1': TwoHidden1h1(),
        '2h10_6k': TwoHidden10_6k(),
        '2h10_5k': TwoHidden10_5k(),
        '2h10_4k': TwoHidden10_4k(),
        '2h10_3k': TwoHidden10_3k(),
        '2h10_2k': TwoHidden10_2k(),
        '2h10_1k': TwoHidden10_1k(),
        '2h10_5h': TwoHidden10_5h(),
        '2h10_1h': TwoHidden10_1h(),
        '2h10_10': TwoHidden10_10(),
        '2h10_1': TwoHidden1_1(),
        '2h1_6k': TwoHidden1_6k(),
        '2h1_5k': TwoHidden1_5k(),
        '2h1_4k': TwoHidden1_4k(),
        '2h1_3k': TwoHidden1_3k(),
        '2h1_2k': TwoHidden1_2k(),
        '2h1_1k': TwoHidden1_1k(),
        '2h1_5h': TwoHidden1_5h(),
        '2h1_1h': TwoHidden1_1h(),
        '2h1_10': TwoHidden1_10(),
        '2h1_1': TwoHidden1_1(),
    }.get(x, OneHidden6k())


if __name__ == "__main__":
    main()

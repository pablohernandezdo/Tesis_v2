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
    Path(f"../Analysis/California_test/{args.model_name}/seismic").mkdir(parents=True, exist_ok=True)
    Path(f"../Analysis/California_test/{args.model_name}/non_seismic").mkdir(parents=True, exist_ok=True)

    # Select training device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Load parameters from trained model
    net.load_state_dict(torch.load('../../models/' + args.model_folder + '/' + args.model_name + '.pth'))
    net.eval()

    # Threshold value
    threshold = 0.95

    # # Load California dataset file
    f = scipy.io.loadmat('../../../Data/California/FSE-06_480SecP_SingDec_StepTest (1).mat')

    # Init figure
    plt.figure()

    # Read data
    traces = f['singdecmatrix']
    traces = traces.transpose()

    # For every trace in the file
    with tqdm.tqdm(total=len(traces), desc='California dataset evaluation') as california_bar:
        idx = 0
        for tr in traces:
            # Resample
            tr = signal.resample(tr, 41228)
            tr = tr[:36000]
            tr = np.reshape(tr, (-1, 6000))

            for trace in tr:
                # Normalize
                trace = trace / np.max(np.abs(trace))

                # trace 2 plot
                trtp = trace

                # Numpy to Torch
                trace = torch.from_numpy(trace).to(device).unsqueeze(0)

                # Prediction
                out_trace = net(trace.float())
                pred_trace = (out_trace > threshold)

                plt.clf()
                if pred_trace:
                    plt.plot(trtp)
                    plt.title(f'Valor de salida: {out_trace.item()}')
                    plt.xlabel('Muestras')
                    plt.ylabel('Amplitud')
                    plt.grid(True)
                    plt.ylim(-1, 1)
                    plt.savefig(f'../Analysis/California_test/{args.model_name}'
                                f'/seismic/{str(out_trace.item())[:5].replace(".","")}_{idx}.png')

                else:
                    plt.plot(trtp)
                    plt.title(f'Valor de salida: {out_trace.item()}')
                    plt.xlabel('Muestras')
                    plt.ylabel('Amplitud')
                    plt.grid(True)
                    plt.ylim(-1, 1)
                    plt.savefig(f'../Analysis/California_test/{args.model_name}'
                                f'/non_seismic/{str(out_trace.item())[:5].replace(".","")}_{idx}.png')

                idx += 1

            california_bar.update()


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
    if x == 'Cnn1_6k':
        return Cnn1_6k()
    if x == 'Cnn1_5k':
        return Cnn1_5k()
    if x == 'Cnn1_4k':
        return Cnn1_4k()
    if x == 'Cnn1_3k':
        return Cnn1_3k()
    if x == 'Cnn1_2k':
        return Cnn1_2k()
    if x == 'Cnn1_1k':
        return Cnn1_1k()
    if x == 'Cnn1_5h':
        return Cnn1_5h()
    if x == 'Cnn1_2h':
        return Cnn1_2h()
    if x == 'Cnn1_5h':
        return Cnn1_1h()
    if x == 'Cnn1_10':
        return Cnn1_10()
    if x == 'Cnn1_6k_6k':
        return Cnn1_6k_6k()
    if x == 'Cnn1_6k_5k':
        return Cnn1_6k_5k()
    if x == 'Cnn1_6k_4k':
        return Cnn1_6k_4k()
    if x == 'Cnn1_6k_3k':
        return Cnn1_6k_3k()
    if x == 'Cnn1_6k_2k':
        return Cnn1_6k_2k()
    if x == 'Cnn1_6k_1k':
        return Cnn1_6k_1k()
    if x == 'Cnn1_6k_5h':
        return Cnn1_6k_5h()
    if x == 'Cnn1_6k_2h':
        return Cnn1_6k_2h()
    if x == 'Cnn1_6k_1h':
        return Cnn1_6k_1h()
    if x == 'Cnn1_6k_10':
        return Cnn1_6k_10()
    if x == 'Cnn1_5k_6k':
        return Cnn1_5k_6k()
    if x == 'Cnn1_5k_5k':
        return Cnn1_5k_5k()
    if x == 'Cnn1_5k_4k':
        return Cnn1_5k_4k()
    if x == 'Cnn1_5k_3k':
        return Cnn1_5k_3k()
    if x == 'Cnn1_5k_2k':
        return Cnn1_5k_2k()
    if x == 'Cnn1_5k_1k':
        return Cnn1_5k_1k()
    if x == 'Cnn1_5k_5h':
        return Cnn1_5k_5h()
    if x == 'Cnn1_5k_2h':
        return Cnn1_5k_2h()
    if x == 'Cnn1_5k_1h':
        return Cnn1_5k_1h()
    if x == 'Cnn1_5k_10':
        return Cnn1_5k_10()
    if x == 'Cnn1_4k_6k':
        return Cnn1_4k_6k()
    if x == 'Cnn1_4k_5k':
        return Cnn1_4k_5k()
    if x == 'Cnn1_4k_4k':
        return Cnn1_4k_4k()
    if x == 'Cnn1_4k_3k':
        return Cnn1_4k_3k()
    if x == 'Cnn1_4k_2k':
        return Cnn1_4k_2k()
    if x == 'Cnn1_4k_1k':
        return Cnn1_4k_1k()
    if x == 'Cnn1_4k_5h':
        return Cnn1_4k_5h()
    if x == 'Cnn1_4k_2h':
        return Cnn1_4k_2h()
    if x == 'Cnn1_4k_1h':
        return Cnn1_4k_1h()
    if x == 'Cnn1_4k_10':
        return Cnn1_4k_10()
    if x == 'Cnn1_3k_6k':
        return Cnn1_3k_6k()
    if x == 'Cnn1_3k_5k':
        return Cnn1_3k_5k()
    if x == 'Cnn1_3k_4k':
        return Cnn1_3k_4k()
    if x == 'Cnn1_3k_3k':
        return Cnn1_3k_3k()
    if x == 'Cnn1_3k_2k':
        return Cnn1_3k_2k()
    if x == 'Cnn1_3k_1k':
        return Cnn1_3k_1k()
    if x == 'Cnn1_3k_5h':
        return Cnn1_3k_5h()
    if x == 'Cnn1_3k_2h':
        return Cnn1_3k_2h()
    if x == 'Cnn1_3k_1h':
        return Cnn1_3k_1h()
    if x == 'Cnn1_3k_10':
        return Cnn1_3k_10()
    if x == 'Cnn1_2k_6k':
        return Cnn1_2k_6k()
    if x == 'Cnn1_2k_5k':
        return Cnn1_2k_5k()
    if x == 'Cnn1_2k_4k':
        return Cnn1_2k_4k()
    if x == 'Cnn1_2k_3k':
        return Cnn1_2k_3k()
    if x == 'Cnn1_2k_2k':
        return Cnn1_2k_2k()
    if x == 'Cnn1_2k_1k':
        return Cnn1_2k_1k()
    if x == 'Cnn1_2k_5h':
        return Cnn1_2k_5h()
    if x == 'Cnn1_2k_2h':
        return Cnn1_2k_2h()
    if x == 'Cnn1_2k_1h':
        return Cnn1_2k_1h()
    if x == 'Cnn1_2k_10':
        return Cnn1_2k_10()
    if x == 'Cnn1_1k_6k':
        return Cnn1_1k_6k()
    if x == 'Cnn1_1k_5k':
        return Cnn1_1k_5k()
    if x == 'Cnn1_1k_4k':
        return Cnn1_1k_4k()
    if x == 'Cnn1_1k_3k':
        return Cnn1_1k_3k()
    if x == 'Cnn1_1k_2k':
        return Cnn1_1k_2k()
    if x == 'Cnn1_1k_1k':
        return Cnn1_1k_1k()
    if x == 'Cnn1_1k_5h':
        return Cnn1_1k_5h()
    if x == 'Cnn1_1k_2h':
        return Cnn1_1k_2h()
    if x == 'Cnn1_1k_1h':
        return Cnn1_1k_1h()
    if x == 'Cnn1_1k_10':
        return Cnn1_1k_10()
    if x == 'Cnn1_5h_6k':
        return Cnn1_5h_6k()
    if x == 'Cnn1_5h_5k':
        return Cnn1_5h_5k()
    if x == 'Cnn1_5h_4k':
        return Cnn1_5h_4k()
    if x == 'Cnn1_5h_3k':
        return Cnn1_5h_3k()
    if x == 'Cnn1_5h_2k':
        return Cnn1_5h_2k()
    if x == 'Cnn1_5h_1k':
        return Cnn1_5h_1k()
    if x == 'Cnn1_5h_5h':
        return Cnn1_5h_5h()
    if x == 'Cnn1_5h_2h':
        return Cnn1_5h_2h()
    if x == 'Cnn1_5h_1h':
        return Cnn1_5h_1h()
    if x == 'Cnn1_5h_10':
        return Cnn1_5h_10()
    if x == 'Cnn1_2h_6k':
        return Cnn1_2h_6k()
    if x == 'Cnn1_2h_5k':
        return Cnn1_2h_5k()
    if x == 'Cnn1_2h_4k':
        return Cnn1_2h_4k()
    if x == 'Cnn1_2h_3k':
        return Cnn1_2h_3k()
    if x == 'Cnn1_2h_2k':
        return Cnn1_2h_2k()
    if x == 'Cnn1_2h_1k':
        return Cnn1_2h_1k()
    if x == 'Cnn1_2h_5h':
        return Cnn1_2h_5h()
    if x == 'Cnn1_2h_2h':
        return Cnn1_2h_2h()
    if x == 'Cnn1_2h_1h':
        return Cnn1_2h_1h()
    if x == 'Cnn1_2h_10':
        return Cnn1_2h_10()
    if x == 'Cnn1_1h_6k':
        return Cnn1_1h_6k()
    if x == 'Cnn1_1h_5k':
        return Cnn1_1h_5k()
    if x == 'Cnn1_1h_4k':
        return Cnn1_1h_4k()
    if x == 'Cnn1_1h_3k':
        return Cnn1_1h_3k()
    if x == 'Cnn1_1h_2k':
        return Cnn1_1h_2k()
    if x == 'Cnn1_1h_1k':
        return Cnn1_1h_1k()
    if x == 'Cnn1_1h_5h':
        return Cnn1_1h_5h()
    if x == 'Cnn1_1h_2h':
        return Cnn1_1h_2h()
    if x == 'Cnn1_1h_1h':
        return Cnn1_1h_1h()
    if x == 'Cnn1_1h_10':
        return Cnn1_1h_10()
    if x == 'Cnn1_10_6k':
        return Cnn1_10_6k()
    if x == 'Cnn1_10_5k':
        return Cnn1_10_5k()
    if x == 'Cnn1_10_4k':
        return Cnn1_10_4k()
    if x == 'Cnn1_10_3k':
        return Cnn1_10_3k()
    if x == 'Cnn1_10_2k':
        return Cnn1_10_2k()
    if x == 'Cnn1_10_1k':
        return Cnn1_10_1k()
    if x == 'Cnn1_10_5h':
        return Cnn1_10_5h()
    if x == 'Cnn1_10_2h':
        return Cnn1_10_2h()
    if x == 'Cnn1_10_1h':
        return Cnn1_10_1h()
    if x == 'Cnn1_10_10':
        return Cnn1_10_10()
    if x == 'Cnn2_6k':
        return Cnn2_6k()
    if x == 'Cnn2_5k':
        return Cnn2_5k()
    if x == 'Cnn2_4k':
        return Cnn2_4k()
    if x == 'Cnn2_3k':
        return Cnn2_3k()
    if x == 'Cnn2_2k':
        return Cnn2_2k()
    if x == 'Cnn2_1k':
        return Cnn2_1k()
    if x == 'Cnn2_5h':
        return Cnn2_5h()
    if x == 'Cnn2_2h':
        return Cnn2_2h()
    if x == 'Cnn2_5h':
        return Cnn2_1h()
    if x == 'Cnn2_10':
        return Cnn2_10()
    if x == 'Cnn2_6k_6k':
        return Cnn2_6k_6k()
    if x == 'Cnn2_6k_5k':
        return Cnn2_6k_5k()
    if x == 'Cnn2_6k_4k':
        return Cnn2_6k_4k()
    if x == 'Cnn2_6k_3k':
        return Cnn2_6k_3k()
    if x == 'Cnn2_6k_2k':
        return Cnn2_6k_2k()
    if x == 'Cnn2_6k_1k':
        return Cnn2_6k_1k()
    if x == 'Cnn2_6k_5h':
        return Cnn2_6k_5h()
    if x == 'Cnn2_6k_2h':
        return Cnn2_6k_2h()
    if x == 'Cnn2_6k_1h':
        return Cnn2_6k_1h()
    if x == 'Cnn2_6k_10':
        return Cnn2_6k_10()
    if x == 'Cnn2_5k_6k':
        return Cnn2_5k_6k()
    if x == 'Cnn2_5k_5k':
        return Cnn2_5k_5k()
    if x == 'Cnn2_5k_4k':
        return Cnn2_5k_4k()
    if x == 'Cnn2_5k_3k':
        return Cnn2_5k_3k()
    if x == 'Cnn2_5k_2k':
        return Cnn2_5k_2k()
    if x == 'Cnn2_5k_1k':
        return Cnn2_5k_1k()
    if x == 'Cnn2_5k_5h':
        return Cnn2_5k_5h()
    if x == 'Cnn2_5k_2h':
        return Cnn2_5k_2h()
    if x == 'Cnn2_5k_1h':
        return Cnn2_5k_1h()
    if x == 'Cnn2_5k_10':
        return Cnn2_5k_10()
    if x == 'Cnn2_4k_6k':
        return Cnn2_4k_6k()
    if x == 'Cnn2_4k_5k':
        return Cnn2_4k_5k()
    if x == 'Cnn2_4k_4k':
        return Cnn2_4k_4k()
    if x == 'Cnn2_4k_3k':
        return Cnn2_4k_3k()
    if x == 'Cnn2_4k_2k':
        return Cnn2_4k_2k()
    if x == 'Cnn2_4k_1k':
        return Cnn2_4k_1k()
    if x == 'Cnn2_4k_5h':
        return Cnn2_4k_5h()
    if x == 'Cnn2_4k_2h':
        return Cnn2_4k_2h()
    if x == 'Cnn2_4k_1h':
        return Cnn2_4k_1h()
    if x == 'Cnn2_4k_10':
        return Cnn2_4k_10()
    if x == 'Cnn2_3k_6k':
        return Cnn2_3k_6k()
    if x == 'Cnn2_3k_5k':
        return Cnn2_3k_5k()
    if x == 'Cnn2_3k_4k':
        return Cnn2_3k_4k()
    if x == 'Cnn2_3k_3k':
        return Cnn2_3k_3k()
    if x == 'Cnn2_3k_2k':
        return Cnn2_3k_2k()
    if x == 'Cnn2_3k_1k':
        return Cnn2_3k_1k()
    if x == 'Cnn2_3k_5h':
        return Cnn2_3k_5h()
    if x == 'Cnn2_3k_2h':
        return Cnn2_3k_2h()
    if x == 'Cnn2_3k_1h':
        return Cnn2_3k_1h()
    if x == 'Cnn2_3k_10':
        return Cnn2_3k_10()
    if x == 'Cnn2_2k_6k':
        return Cnn2_2k_6k()
    if x == 'Cnn2_2k_5k':
        return Cnn2_2k_5k()
    if x == 'Cnn2_2k_4k':
        return Cnn2_2k_4k()
    if x == 'Cnn2_2k_3k':
        return Cnn2_2k_3k()
    if x == 'Cnn2_2k_2k':
        return Cnn2_2k_2k()
    if x == 'Cnn2_2k_1k':
        return Cnn2_2k_1k()
    if x == 'Cnn2_2k_5h':
        return Cnn2_2k_5h()
    if x == 'Cnn2_2k_2h':
        return Cnn2_2k_2h()
    if x == 'Cnn2_2k_1h':
        return Cnn2_2k_1h()
    if x == 'Cnn2_2k_10':
        return Cnn2_2k_10()
    if x == 'Cnn2_1k_6k':
        return Cnn2_1k_6k()
    if x == 'Cnn2_1k_5k':
        return Cnn2_1k_5k()
    if x == 'Cnn2_1k_4k':
        return Cnn2_1k_4k()
    if x == 'Cnn2_1k_3k':
        return Cnn2_1k_3k()
    if x == 'Cnn2_1k_2k':
        return Cnn2_1k_2k()
    if x == 'Cnn2_1k_1k':
        return Cnn2_1k_1k()
    if x == 'Cnn2_1k_5h':
        return Cnn2_1k_5h()
    if x == 'Cnn2_1k_2h':
        return Cnn2_1k_2h()
    if x == 'Cnn2_1k_1h':
        return Cnn2_1k_1h()
    if x == 'Cnn2_1k_10':
        return Cnn2_1k_10()
    if x == 'Cnn2_5h_6k':
        return Cnn2_5h_6k()
    if x == 'Cnn2_5h_5k':
        return Cnn2_5h_5k()
    if x == 'Cnn2_5h_4k':
        return Cnn2_5h_4k()
    if x == 'Cnn2_5h_3k':
        return Cnn2_5h_3k()
    if x == 'Cnn2_5h_2k':
        return Cnn2_5h_2k()
    if x == 'Cnn2_5h_1k':
        return Cnn2_5h_1k()
    if x == 'Cnn2_5h_5h':
        return Cnn2_5h_5h()
    if x == 'Cnn2_5h_2h':
        return Cnn2_5h_2h()
    if x == 'Cnn2_5h_1h':
        return Cnn2_5h_1h()
    if x == 'Cnn2_5h_10':
        return Cnn2_5h_10()
    if x == 'Cnn2_2h_6k':
        return Cnn2_2h_6k()
    if x == 'Cnn2_2h_5k':
        return Cnn2_2h_5k()
    if x == 'Cnn2_2h_4k':
        return Cnn2_2h_4k()
    if x == 'Cnn2_2h_3k':
        return Cnn2_2h_3k()
    if x == 'Cnn2_2h_2k':
        return Cnn2_2h_2k()
    if x == 'Cnn2_2h_1k':
        return Cnn2_2h_1k()
    if x == 'Cnn2_2h_5h':
        return Cnn2_2h_5h()
    if x == 'Cnn2_2h_2h':
        return Cnn2_2h_2h()
    if x == 'Cnn2_2h_1h':
        return Cnn2_2h_1h()
    if x == 'Cnn2_2h_10':
        return Cnn2_2h_10()
    if x == 'Cnn2_1h_6k':
        return Cnn2_1h_6k()
    if x == 'Cnn2_1h_5k':
        return Cnn2_1h_5k()
    if x == 'Cnn2_1h_4k':
        return Cnn2_1h_4k()
    if x == 'Cnn2_1h_3k':
        return Cnn2_1h_3k()
    if x == 'Cnn2_1h_2k':
        return Cnn2_1h_2k()
    if x == 'Cnn2_1h_1k':
        return Cnn2_1h_1k()
    if x == 'Cnn2_1h_5h':
        return Cnn2_1h_5h()
    if x == 'Cnn2_1h_2h':
        return Cnn2_1h_2h()
    if x == 'Cnn2_1h_1h':
        return Cnn2_1h_1h()
    if x == 'Cnn2_1h_10':
        return Cnn2_1h_10()
    if x == 'Cnn2_10_6k':
        return Cnn2_10_6k()
    if x == 'Cnn2_10_5k':
        return Cnn2_10_5k()
    if x == 'Cnn2_10_4k':
        return Cnn2_10_4k()
    if x == 'Cnn2_10_3k':
        return Cnn2_10_3k()
    if x == 'Cnn2_10_2k':
        return Cnn2_10_2k()
    if x == 'Cnn2_10_1k':
        return Cnn2_10_1k()
    if x == 'Cnn2_10_5h':
        return Cnn2_10_5h()
    if x == 'Cnn2_10_2h':
        return Cnn2_10_2h()
    if x == 'Cnn2_10_1h':
        return Cnn2_10_1h()
    if x == 'Cnn2_10_10':
        return Cnn2_10_10()
    else:
        return Cnn2_10_10()


if __name__ == "__main__":
    main()

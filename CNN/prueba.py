import torch
import argparse
import scipy.io
import numpy as np

from scipy import signal
from scipy.signal import butter, lfilter

from model import *


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Default_model', help="Classifier model path")
    parser.add_argument("--classifier", default='C', help="Choose classifier architecture")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Load parameters from trained model
    net.load_state_dict(torch.load('../../ANN/models/' + args.model_name + '.pth'))
    net.eval()

    # AQUÍ DEFINIR LA SEÑAL QUE SE QUIERE PROBAR, DEBE TENER 6000 MUESTRAS
    senal = np.ones((1, 6000))

    senal = torch.from_numpy(senal).to(device)

    # Prediccion
    out = net(senal.float())
    out = out.data.item()

    # Results
    print(f'Resultado inferencia: {out}\n')


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
        'C': Classifier(),
        'S': Classifier_S(),
        'XS': Classifier_XS(),
        'XL': Classifier_XL(),
        'XXL':Classifier_XXL(),
        'XXXL': Classifier_XXXL(),
    }.get(x, Classifier())


if __name__ == "__main__":
    main()

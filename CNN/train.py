import time
import argparse

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from humanfriendly import format_timespan

from model import *
from dataset import HDF5Dataset


def main():
    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Default_model', help="Name of model to save")
    parser.add_argument("--classifier", default='C', help="Choose classifier architecture, C, CBN")
    parser.add_argument("--train_path", default='Train_data.hdf5', help="HDF5 train Dataset path")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.00001, help="Adam learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay parameter")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train dataset
    train_dataset = HDF5Dataset(args.train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Count number of parameters
    nparams = count_parameters(net)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.wd)

    # Start training
    with tqdm.tqdm(total=args.n_epochs, desc='Epochs') as epoch_bar:
        for epoch in range(args.n_epochs):

            total_loss = 0

            with tqdm.tqdm(total=len(train_loader), desc='Batches', leave=False) as batch_bar:
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()

                    outputs = net(inputs)
                    loss = criterion(outputs, labels.float())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    batch_bar.update()
                epoch_bar.update()

    # Save model
    torch.save(net.state_dict(), '../models/' + args.model_name + '.pth')

    # Measure training, and execution times
    end_tm = time.time()

    # Training time
    tr_t = end_tm - start_time

    print(f'Execution details: \n{args}\n'
          f'Number of parameters: {nparams}\n'
          f'Training time: {format_timespan(tr_t)}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_classifier(x):
    return {
        '1c1h': CNN1P1H1h(),
        '1c2h': CNN1P1H2h(),
        '1c5h': CNN1P1H5h(),
        '1c1k': CNN1P1H1k(),
        '1c2k': CNN1P1H2k(),
        '1c3k': CNN1P1H3k(),
        '1c4k': CNN1P1H4k(),
        '1c5k': CNN1P1H5k(),
        '1c6k': CNN1P1H6k(),
        '1c10k': CNN1P1H10k(),
        '1c10k10k': CNN1P2H10k10k(),
        '1c10k5k': CNN1P2H10k5k(),
        '1c10k1k': CNN1P2H10k1k(),
        '1c10k1h': CNN1P2H10k1h(),
        '1c10k10': CNN1P2H10k10(),
        '1c6k6k': CNN1P2H6k6k(),
        '1c6k1k': CNN1P2H6k1k(),
        '1c6k1h': CNN1P2H6k1h(),
        '1c6k10': CNN1P2H6k10(),
        '1c5k5k': CNN1P2H5k5k(),
        '1c5k1k': CNN1P2H5k1k(),
        '1c5k1h': CNN1P2H5k1h(),
        '1c5k10': CNN1P2H5k10(),
        '1c4k4k': CNN1P2H4k4k(),
        '1c4k1k': CNN1P2H4k1k(),
        '1c4k1h': CNN1P2H4k1h(),
        '1c4k10': CNN1P2H4k10(),
        '1c3k3k': CNN1P2H3k3k(),
        '1c3k1k': CNN1P2H3k1k(),
        '1c3k1h': CNN1P2H3k1h(),
        '1c3k10': CNN1P2H3k10(),
        '1c2k2k': CNN1P2H2k2k(),
        '1c2k1k': CNN1P2H2k1k(),
        '1c2k1h': CNN1P2H2k1h(),
        '1c2k10': CNN1P2H2k10(),
        '1c1k1k': CNN1P2H1k1k(),
        '1c1k1h': CNN1P2H1k1h(),
        '1c1k10': CNN1P2H1k10(),
        '1c5h5h': CNN1P2H5h5h(),
        '1c5h1h': CNN1P2H5h1h(),
        '1c5h10': CNN1P2H5h10(),
        '1c2h2h': CNN1P2H2h2h(),
        '1c2h1h': CNN1P2H2h1h(),
        '1c2h10': CNN1P2H2h10(),
        '1c1h1h': CNN1P2H1h1h(),
        '1c1h10': CNN1P2H1h10(),
        '2c20k': CNN2P1H20k(),
        '2c15k': CNN2P1H15k(),
        '2c10k': CNN2P1H10k(),
        '2c5k': CNN2P1H5k(),
        '2c3k': CNN2P1H3k(),
        '2c2k': CNN2P1H2k(),
        '2c1k': CNN2P1H1k(),
        '2c20k20k': CNN2P1H20k20k(),
        '2c20k10k': CNN2P1H20k10k(),
        '2c20k5k': CNN2P1H20k5k(),
        '2c20k2k': CNN2P1H20k2k(),
        '2c20k1k': CNN2P1H20k1k(),
        '2c20k5h': CNN2P1H20k5h(),
        '2c20k1h': CNN2P1H20k1h(),
        '2c20k10': CNN2P1H20k10(),
        '2c15k15k': CNN2P1H15k15k(),
        '2c15k10k': CNN2P1H15k10k(),
        '2c15k5k': CNN2P1H15k5k(),
        '2c15k2k': CNN2P1H15k2k(),
        '2c15k1k': CNN2P1H15k1k(),
        '2c15k5h': CNN2P1H15k5h(),
        '2c15k1h': CNN2P1H15k1h(),
        '2c15k10': CNN2P1H15k10(),
        '2c10k10k': CNN2P1H10k10k(),
        '2c10k5k': CNN2P1H10k5k(),
        '2c10k2k': CNN2P1H10k2k(),
        '2c10k1k': CNN2P1H10k1k(),
        '2c10k5h': CNN2P1H10k5h(),
        '2c10k1h': CNN2P1H10k1h(),
        '2c10k10': CNN2P1H10k10(),
        '2c5k5k': CNN2P1H5k5k(),
        '2c5k2k': CNN2P1H5k2k(),
        '2c5k1k': CNN2P1H5k1k(),
        '2c5k5h': CNN2P1H5k5h(),
        '2c5k1h': CNN2P1H5k1h(),
        '2c5k10': CNN2P1H5k10(),
        '2c3k3k': CNN2P1H3k3k(),
        '2c3k2k': CNN2P1H3k2k(),
        '2c3k1k5': CNN2P1H3k1k5(),
        '2c3k1k': CNN2P1H3k1k(),
        '2c3k5h': CNN2P1H3k5h(),
        '2c3k1h': CNN2P1H3k1h(),
        '2c3k10': CNN2P1H3k10(),
        '2c2k2k': CNN2P1H2k2k(),
        '2c2k1k': CNN2P1H2k1k(),
        '2c2k5h': CNN2P1H2k5h(),
        '2c2k1h': CNN2P1H2k1h(),
        '2c2k10': CNN2P1H2k10(),
        '2c1k1k': CNN2P1H1k1k(),
        '2c1k5h': CNN2P1H1k5h(),
        '2c1k1h': CNN2P1H1k1h(),
        '2c1k10': CNN2P1H1k10(),
    }.get(x, 'hola')


if __name__ == "__main__":
    main()

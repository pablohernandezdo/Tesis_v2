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


def get_classifier(x):
    return {
        '1C1h': CNN1P1H1h(),
        '1C2h': CNN1P1H2h(),
        '1C5h': CNN1P1H5h(),
        '1C1K': CNN1P1H1k(),
        '1C2K': CNN1P1H2k(),
        '1C3K': CNN1P1H3k(),
        '1C4K': CNN1P1H4k(),
        '1C5K': CNN1P1H5k(),
        '1C6K': CNN1P1H6k(),
        '1C10K': CNN1P1H10k(),
        '1C10k10k': CNN1P2H10k10k(),
        '1C10k5k': CNN1P2H10k5k(),
        '1C10k1k': CNN1P2H10k1k(),
        '1C10k1h': CNN1P2H10k1h(),
        '1C10k10': CNN1P2H10k10(),
        '1C6k6k': CNN1P2H6k6k(),
        '1C6k1k': CNN1P2H6k1k(),
        '1C6k1h': CNN1P2H6k1h(),
        '1C6k10': CNN1P2H6k10(),
        '1C5k5k': CNN1P2H5k5k(),
        '1C5k1k': CNN1P2H5k1k(),
        '1C5k1h': CNN1P2H5k1h(),
        '1C5k10': CNN1P2H5k10(),
        '1C4k4k': CNN1P2H4k4k(),
        '1C4k1k': CNN1P2H4k1k(),
        '1C4k1h': CNN1P2H4k1h(),
        '1C4k10': CNN1P2H4k10(),
        '1C3k3k': CNN1P2H3k3k(),
        '1C3k1k': CNN1P2H3k1k(),
        '1C3k1h': CNN1P2H3k1h(),
        '1C3k10': CNN1P2H3k10(),
        '1C2k2k': CNN1P2H2k2k(),
        '1C2k1k': CNN1P2H2k1k(),
        '1C2k1h': CNN1P2H2k1h(),
        '1C2k10': CNN1P2H2k10(),
        '1C1k1k': CNN1P2H1k1k(),
        '1C1k1h': CNN1P2H1k1h(),
        '1C1k10': CNN1P2H1k10(),
        '1C5h5h': CNN1P2H5h5h(),
        '1C5h1h': CNN1P2H5h1h(),
        '1C5h10': CNN1P2H5h10(),
        '1C2h2h': CNN1P2H2h2h(),
        '1C2h1h': CNN1P2H2h1h(),
        '1C2h10': CNN1P2H2h10(),
        '1C1h1h': CNN1P2H1h1h(),
        '1C1h10': CNN1P2H1h10(),
        '2C20k': CNN2P1H20K(),
        '2C15k': CNN2P1H15K(),
        '2C10k': CNN2P1H10K(),
        '2C5k': CNN2P1H5K(),
        '2C3k': CNN2P1H3K(),
        '2C2k': CNN2P1H2K(),
        '2C1k': CNN2P1H1K(),
        '2C20k20k': CNN2P1H20k20k(),
        '2C20k10k': CNN2P1H20k10k(),
        '2C20k5k': CNN2P1H20k5k(),
        '2C20k2k': CNN2P1H20k2k(),
        '2C20k1k': CNN2P1H20k1k(),
        '2C20k5h': CNN2P1H20k5h(),
        '2C20k1h': CNN2P1H20k1h(),
        '2C20k10': CNN2P1H20k10(),
        '2C15k15k': CNN2P1H15k15k(),
        '2C15k10k': CNN2P1H15k10k(),
        '2C15k5k': CNN2P1H15k5k(),
        '2C15k2k': CNN2P1H15k2k(),
        '2C15k1k': CNN2P1H15k1k(),
        '2C15k5h': CNN2P1H15k5h(),
        '2C15k1h': CNN2P1H15k1h(),
        '2C15k10': CNN2P1H15k10(),
        '2C10k10k': CNN2P1H10k10k(),
        '2C10k5k': CNN2P1H10k5k(),
        '2C10k2k': CNN2P1H10k2k(),
        '2C10k1k': CNN2P1H10k1k(),
        '2C10k5h': CNN2P1H10k5h(),
        '2C10k1h': CNN2P1H10k1h(),
        '2C10k10': CNN2P1H10k10(),
        '2C5k5k': CNN2P1H5k5k(),
        '2C5k2k': CNN2P1H5k2k(),
        '2C5k1k': CNN2P1H5k1k(),
        '2C5k5h': CNN2P1H5k5h(),
        '2C5k1h': CNN2P1H5k1h(),
        '2C5k10': CNN2P1H5k10(),
        '2C3k3k': CNN2P1H3k3k(),
        '2C3k2k': CNN2P1H3k2k(),
        '2C3k1k5': CNN2P1H3k1k5(),
        '2C3k1k': CNN2P1H3k1k(),
        '2C3k5h': CNN2P1H3k5h(),
        '2C3k1h': CNN2P1H3k1h(),
        '2C3k10': CNN2P1H3k10(),
        '2C2k2k': CNN2P1H2k2k(),
        '2C2k1k': CNN2P1H2k1k(),
        '2C2k5h': CNN2P1H2k5h(),
        '2C2k1h': CNN2P1H2k1h(),
        '2C2k10': CNN2P1H2k10(),
        '2C1k1k': CNN2P1H1k1k(),
        '2C1k5h': CNN2P1H1k5h(),
        '2C1k1h': CNN2P1H1k1h(),
        '2C1k10': CNN2P1H1k10(),
        'C': ClassConv(),
        'CBN': ClassConvBN(),
        'CBN_v2': CBN_v2(),
    }.get(x, ClassConv())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()

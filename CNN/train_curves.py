import time
import tqdm
import argparse
import matplotlib.pyplot as plt

from humanfriendly import format_timespan

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import HDF5Dataset
from torch.utils.data import DataLoader

from model import *

from torch.utils.tensorboard import SummaryWriter


def main():
    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Default_model', help="Name of model to save")
    parser.add_argument("--classifier", default='C', help="Choose classifier architecture, C, CBN")
    parser.add_argument("--train_path", default='Train_data.hdf5', help="HDF5 train Dataset path")
    parser.add_argument("--val_path", default='Validation_data.hdf5', help="HDF5 validation Dataset path")
    parser.add_argument("--test_path", default='Test_data.hdf5', help="HDF5 test Dataset path")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.00001, help="Adam learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay parameter")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
    args = parser.parse_args()

    print(f'Execution details: \n {args}')

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Start tensorboard SummaryWriter
    tb = SummaryWriter('../runs/Seismic')

    # Train dataset
    train_dataset = HDF5Dataset(args.train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Validation dataset
    val_dataset = HDF5Dataset(args.val_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Load specified Classifier
    if args.classifier == 'CBN':
        net = ClassConvBN()
    elif args.classifier == 'CBN_v2':
        net = CBN_v2()
    elif args.classifier == 'C':
        net = ClassConv()
    else:
        net = ClassConv()
        print('Bad Classifier option, running classifier C')
    net.to(device)

    # Add model graph to tensorboard
    # images, labels = next(iter(train_loader))
    # images, labels = images.to(device), labels.to(device)
    # tb.add_graph(net, images)

    # Loss function and optimizer
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.wd)

    # Loss id for tensorboard logs
    loss_id = 0

    # Training and validation errors
    train_error = []
    val_error = []

    # Start training
    with tqdm.tqdm(total=args.n_epochs, desc='Epochs') as epoch_bar:
        for epoch in range(args.n_epochs):

            total_loss = 0

            with tqdm.tqdm(total=len(train_loader), desc='Batches', leave=False) as batch_bar:
                for i, data in enumerate(train_loader):

                    # Network to train mode
                    net.train()

                    inputs, labels = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()

                    outputs = net(inputs)
                    tb.add_scalar('Output', outputs[0].item(), loss_id)
                    loss = criterion(outputs, labels.float())
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    loss_id += 1

                    tb.add_scalar('Loss', loss.item(), loss_id)
                    batch_bar.update()

                tb.add_scalar('Total_Loss', total_loss, epoch)
                epoch_bar.update()

            # Network to evaluation mode
            net.eval()

            with torch.no_grad():
                # Training error
                train_total = 0
                train_correct = 0

                for data in train_loader:
                    traces, labels = data[0].to(device), data[1].to(device)
                    outputs = net(traces)
                    predicted = torch.round(outputs)

                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                train_error.append(train_correct / train_total)

                # Validation error
                val_total = 0
                val_correct = 0

                for data in val_loader:
                    traces, labels = data[0].to(device), data[1].to(device)
                    outputs = net(traces)
                    predicted = torch.round(outputs)

                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                val_error.append(val_correct / val_total)

    # Close tensorboard
    tb.close()

    # Save model
    torch.save(net.state_dict(), '../models/' + args.model_name + '.pth')

    # Measure training, and execution times
    end_tm = time.time()

    # Training time
    tr_t = end_tm - start_time

    print(f'Training time: {format_timespan(tr_t)}')

    print(f'Train error: {train_error:5.3f}\n'
          f'Val error: {val_error:5.3f}')
    # plt.figure()
    # plt.plot(train_error, 'r')
    # plt.plot(val_error, 'b')
    # plt.savefig('ERRORS.png')


if __name__ == "__main__":
    main()

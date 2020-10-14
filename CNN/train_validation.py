import time
import argparse
from pathlib import Path

import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from humanfriendly import format_timespan

from model import *
from dataset import HDF5Dataset


def main():
    # Create learning curves folder
    Path("../Learning_curves/Accuracy").mkdir(exist_ok=True, parents=True)
    Path("../Learning_curves/Loss").mkdir(exist_ok=True)

    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Default_model', help="Name of model to save")
    parser.add_argument("--classifier", default='C', help="Choose classifier architecture, C, CBN")
    parser.add_argument("--train_path", default='Train_data.hdf5', help="HDF5 train Dataset path")
    parser.add_argument("--val_path", default='Validation_data.hdf5', help="HDF5 validation Dataset path")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
    parser.add_argument("--eval_iter", type=int, default=20, help="Number of batches between validations")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
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

    # Validation dataset
    val_dataset = HDF5Dataset(args.val_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Count number of parameters
    nparams = count_parameters(net)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.wd)

    # Training and validation errors
    tr_accuracies = []
    val_accuracies = []

    # Training and validation losses
    tr_losses = []
    val_losses = []

    # Early stopping
    val_acc = 1
    earlys = np.zeros(args.patience).tolist()

    # Start training
    with tqdm.tqdm(total=args.n_epochs, desc='Epochs') as epoch_bar:
        for epoch in range(args.n_epochs):

            n_correct, n_total = 0, 0

            # Early stopping
            if all(val_acc <= i for i in earlys):
                print('Early stopping training')
                break

            with tqdm.tqdm(total=len(train_loader), desc='Batches', leave=False) as batch_bar:
                for i, data in enumerate(train_loader):

                    # Network to train mode
                    net.train()

                    # Clear gradient accumulators
                    optimizer.zero_grad()

                    # Get batch data and labels
                    inputs, labels = data[0].to(device), data[1].to(device)

                    # Forward pass
                    outputs = net(inputs)

                    # Predicted labels
                    predicted = torch.round(outputs)

                    # Calculate accuracy on current batch
                    n_total += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
                    train_acc = 100 * n_correct / n_total

                    # Calculate loss
                    loss = criterion(outputs, labels.float())

                    # Backpropagation
                    loss.backward()

                    # Optimize
                    optimizer.step()

                    # Check validation accuracy periodically
                    if i % args.eval_iter == 0:
                        # Switch model to eval mode
                        net.eval()

                        # Calculate accuracy on validation
                        total_val_loss = 0
                        total_val, correct_val = 0, 0

                        with torch.no_grad():
                            for val_data in val_loader:

                                # Retrieve data and labels
                                traces, labels = val_data[0].to(device), val_data[1].to(device)

                                # Forward pass
                                outputs = net(traces)

                                # Calculate loss
                                val_loss = criterion(outputs, labels.float())

                                # Total loss for epoch
                                total_val_loss += val_loss.item()

                                # Predicted labels
                                predicted = torch.round(outputs)

                                # Sum up correct and total validation examples
                                total_val += labels.size(0)
                                correct_val += (predicted == labels).sum().item()

                            val_avg_loss = total_val_loss / len(val_loader)

                        # Calculate validation accuracy
                        val_acc = 100 * correct_val / total_val

                        # Save acc for early stopping
                        earlys.pop(0)
                        earlys.append(val_acc)

                    # Save loss to list
                    val_losses.append(val_avg_loss)
                    tr_losses.append(loss)

                    # Append training and validation accuracies
                    tr_accuracies.append(train_acc)
                    val_accuracies.append(val_acc)

                    # Update batch bar
                    batch_bar.update()

                    # Early stopping
                    if all(val_acc <= i for i in earlys):
                        print('Early stopping training')
                        break

                # Update epochs bar
                epoch_bar.update()

    # Save model
    torch.save(net.state_dict(), '../models/' + args.model_name + '.pth')

    # Measure training, and execution times
    end_tm = time.time()

    # Training time
    tr_t = end_tm - start_time

    # Plot train and validation accuracies
    learning_curve_acc(tr_accuracies, val_accuracies, args.model_name)

    # Plot train and validation losses
    learning_curve_loss(tr_losses, val_losses, args.model_name)

    print(f'Execution details: \n{args}\n'
          f'Number of parameters: {nparams}\n'
          f'Training time: {format_timespan(tr_t)}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def learning_curve_acc(tr_acc, val_acc, model_name):
    plt.figure()
    line_tr, = plt.plot(tr_acc, label='Training accuracy')
    line_val, = plt.plot(val_acc, label='Validation accuracy')
    plt.grid(True)
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.legend(handles=[line_tr, line_val], loc='best')
    plt.savefig(f'../Learning_curves/{model_name}_accuracies.png')


def learning_curve_loss(tr_loss, val_loss, model_name):
    plt.figure()
    line_tr, = plt.plot(tr_loss, label='Training Loss')
    line_val, = plt.plot(val_loss, label='Validation Loss')
    plt.grid(True)
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.title(f'Loss learning curve model {model_name}')
    plt.legend(handles=[line_tr, line_val], loc='best')
    plt.savefig(f'../Learning_curves/Loss/{model_name}_Losses.png')


def get_classifier(x):
    return {
        '1C1h': CNN1P1H1h(),
        '1C2h': CNN1P1H2h(),
        '1C5h': CNN1P1H5h(),
        '1C1k': CNN1P1H1k(),
        '1C2k': CNN1P1H2k(),
        '1C3k': CNN1P1H3k(),
        '1C4k': CNN1P1H4k(),
        '1C5k': CNN1P1H5k(),
        '1C6k': CNN1P1H6k(),
        '1C10k': CNN1P1H10k(),
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
    }.get(x, 'hola')


if __name__ == "__main__":
    main()

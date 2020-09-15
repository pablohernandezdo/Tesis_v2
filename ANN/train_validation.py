import time
import argparse
from pathlib import Path

import tqdm
import torch
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
    parser.add_argument("--eval_iter", type=int, default=30, help="Number of batches between validations")
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

    # Start training
    with tqdm.tqdm(total=args.n_epochs, desc='Epochs') as epoch_bar:
        for epoch in range(args.n_epochs):

            total_tr_loss = 0
            n_correct, n_total = 0, 0

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

                    # Calculate total loss
                    # total_tr_loss += loss.item()

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

                    # Save loss to list
                    val_losses.append(val_avg_loss)
                    tr_losses.append(loss)

                    # Append training and validation accuracies
                    tr_accuracies.append(train_acc)
                    val_accuracies.append(val_acc)

                    # Update batch bar
                    batch_bar.update()

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

    # Plot train and validation accuracies
    learning_curve_loss(tr_losses, val_losses, args.model_name)

    print(f'Execution details: \n{args}\n'
          f'Number of parameters: {nparams}\n'
          f'Training time: {format_timespan(tr_t)}')


def get_classifier(x):
    return {
        'C': Classifier(),
        'S': Classifier_S(),
        'XS': Classifier_XS(),
        'XL': Classifier_XL(),
        'XXL':Classifier_XXL(),
        'XXXL': Classifier_XXXL(),
    }.get(x, Classifier())


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
    plt.savefig(f'../Learning_curves/Accuracy/{model_name}_accuracies.png')


def learning_curve_loss(tr_loss, val_loss, model_name):
    plt.figure()
    line_tr, = plt.plot(tr_loss, label='Training Loss')
    line_val, = plt.plot(val_loss, label='Validation Loss')
    plt.grid(True)
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.legend(handles=[line_tr, line_val], loc='best')
    plt.savefig(f'../Learning_curves/Loss/{model_name}_Losses.png')


if __name__ == "__main__":
    main()

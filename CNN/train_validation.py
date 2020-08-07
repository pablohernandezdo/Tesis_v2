import time
import argparse

import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from humanfriendly import format_timespan
from torch.utils.tensorboard import SummaryWriter

from .model import *
from .dataset import HDF5Dataset


def main():
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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.wd)

    # Loss id for tensorboard logs
    # loss_id = 0

    # Training and validation errors
    tr_accuracies = []
    val_accuracies = []

    # Start training
    with tqdm.tqdm(total=args.n_epochs, desc='Epochs') as epoch_bar:
        for epoch in range(args.n_epochs):

            total_loss = 0
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
                    total_loss += loss.item()

                    # loss_id += 1

                    # Check validation accuracy periodically
                    if i % 20 == 0:
                        # Switch model to eval mode
                        net.eval()

                        # Calculate accuracy on validation
                        total_val, correct_val = 0, 0

                        with torch.no_grad():
                            for val_data in val_loader:

                                # Retrieve data and labels
                                traces, labels = val_data[0].to(device), val_data[1].to(device)

                                # Forward pass
                                outputs = net(traces)

                                # Predicted labels
                                predicted = torch.round(outputs)

                                # Sum up correct and total validation examples
                                total_val += labels.size(0)
                                correct_val += (predicted == labels).sum().item()

                        # Calculate validation accuracy
                        val_acc = 100 * correct_val / total_val

                    # Append training and validation accuracies
                    tr_accuracies.append(train_acc)
                    val_accuracies.append(val_acc)

                    # tb.add_scalar('Loss', loss.item(), loss_id)
                    # Update batch bar
                    batch_bar.update()

                # tb.add_scalar('Total_Loss', total_loss, epoch)
                # Update epochs bar
                epoch_bar.update()

    # Close tensorboard
    tb.close()

    # Save model
    torch.save(net.state_dict(), '../models/' + args.model_name + '.pth')

    # Measure training, and execution times
    end_tm = time.time()

    # Training time
    tr_t = end_tm - start_time

    print(f'Training time: {format_timespan(tr_t)}')

    plt.figure()
    line_tr, = plt.plot(tr_accuracies, label='Training accuracy')
    line_val, = plt.plot(val_accuracies, label='Validation accuracy')
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend(handles=[line_tr, line_val], loc='best')
    plt.savefig(f'{args.model_name}_accuracies.png')


if __name__ == "__main__":
    main()

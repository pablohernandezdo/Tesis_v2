import time
import argparse

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from humanfriendly import format_timespan
# from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import HDF5Dataset


def main():
    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='Default_model', help="Name of model to save")
    parser.add_argument("--model_folder", default='default', help="Folder to save model")
    parser.add_argument("--classifier", default='C', help="Choose classifier architecture, C, S, XS, XL, XXL, XXXL")
    parser.add_argument("--train_path", default='Train_data.hdf5', help="HDF5 train Dataset path")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay parameter")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Start tensorboard SummaryWriter
    # tb = SummaryWriter('../runs/Seismic')

    # Train dataset
    train_dataset = HDF5Dataset(args.train_path)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Count number of parameters
    nparams = count_parameters(net)

    # Add model graph to tensorboard
    # traces, labels = next(iter(trainloader))
    # traces, labels = traces.to(device), labels.to(device)
    # tb.add_graph(net, traces)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.wd)

    # Loss id for tensorboard logs
    # loss_id = 0

    # Start training
    with tqdm.tqdm(total=args.n_epochs, desc='Epochs') as epoch_bar:
        for epoch in range(args.n_epochs):

            total_loss = 0

            with tqdm.tqdm(total=len(trainloader), desc='Batches', position=1) as batch_bar:
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)

                    optimizer.zero_grad()

                    outputs = net(inputs)
                    # tb.add_scalar('Output', outputs[0].item(), loss_id)
                    loss = criterion(outputs, labels.float())
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    # loss_id += 1

                    # tb.add_scalar('Loss', loss.item(), loss_id)
                    batch_bar.update()
                # tb.add_scalar('Total_Loss', total_loss, epoch)
                epoch_bar.update()

    # Close tensorboard
    # tb.close()

    # Save model
    torch.save(net.state_dict(), '../models/' + args.model_folder + '/' + args.model_name + '.pth')

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
    if x == 'Lstm_16_16_1_1':
        return Lstm_16_16_1_1()
    if x == 'Lstm_16_16_2_1':
        return Lstm_16_16_2_1()
    if x == 'Lstm_16_16_5_1':
        return Lstm_16_16_5_1()
    if x == 'Lstm_16_16_10_1':
        return Lstm_16_16_10_1()
    if x == 'Lstm_16_16_20_1':
        return Lstm_16_16_20_1()
    if x == 'Lstm_16_32_1_1':
        return Lstm_16_32_1_1()
    if x == 'Lstm_16_32_2_1':
        return Lstm_16_32_2_1()
    if x == 'Lstm_16_32_5_1':
        return Lstm_16_32_5_1()
    if x == 'Lstm_16_32_10_1':
        return Lstm_16_32_10_1()
    if x == 'Lstm_16_32_20_1':
        return Lstm_16_32_20_1()
    if x == 'Lstm_16_64_1_1':
        return Lstm_16_64_1_1()
    if x == 'Lstm_16_64_2_1':
        return Lstm_16_64_2_1()
    if x == 'Lstm_16_64_5_1':
        return Lstm_16_64_5_1()
    if x == 'Lstm_16_64_10_1':
        return Lstm_16_64_10_1()
    if x == 'Lstm_16_64_20_1':
        return Lstm_16_64_20_1()
    if x == 'Lstm_16_128_1_1':
        return Lstm_16_128_1_1()
    if x == 'Lstm_16_128_2_1':
        return Lstm_16_128_2_1()
    if x == 'Lstm_16_128_5_1':
        return Lstm_16_128_5_1()
    if x == 'Lstm_16_128_10_1':
        return Lstm_16_128_10_1()
    if x == 'Lstm_16_128_20_1':
        return Lstm_16_128_20_1()
    if x == 'Lstm_16_256_1_1':
        return Lstm_16_256_1_1()
    if x == 'Lstm_16_256_2_1':
        return Lstm_16_256_2_1()
    if x == 'Lstm_16_256_5_1':
        return Lstm_16_256_5_1()
    if x == 'Lstm_16_256_10_1':
        return Lstm_16_256_10_1()
    if x == 'Lstm_16_256_20_1':
        return Lstm_16_256_20_1()
    if x == 'Lstm_32_16_1_1':
        return Lstm_32_16_1_1()
    if x == 'Lstm_32_16_2_1':
        return Lstm_32_16_2_1()
    if x == 'Lstm_32_16_5_1':
        return Lstm_32_16_5_1()
    if x == 'Lstm_32_16_10_1':
        return Lstm_32_16_10_1()
    if x == 'Lstm_32_16_20_1':
        return Lstm_32_16_20_1()
    if x == 'Lstm_32_32_1_1':
        return Lstm_32_32_1_1()
    if x == 'Lstm_32_32_2_1':
        return Lstm_32_32_2_1()
    if x == 'Lstm_32_32_5_1':
        return Lstm_32_32_5_1()
    if x == 'Lstm_32_32_10_1':
        return Lstm_32_32_10_1()
    if x == 'Lstm_32_32_20_1':
        return Lstm_32_32_20_1()
    if x == 'Lstm_32_64_1_1':
        return Lstm_32_64_1_1()
    if x == 'Lstm_32_64_2_1':
        return Lstm_32_64_2_1()
    if x == 'Lstm_32_64_5_1':
        return Lstm_32_64_5_1()
    if x == 'Lstm_32_64_10_1':
        return Lstm_32_64_10_1()
    if x == 'Lstm_32_64_20_1':
        return Lstm_32_64_20_1()
    if x == 'Lstm_32_128_1_1':
        return Lstm_32_128_1_1()
    if x == 'Lstm_32_128_2_1':
        return Lstm_32_128_2_1()
    if x == 'Lstm_32_128_5_1':
        return Lstm_32_128_5_1()
    if x == 'Lstm_32_128_10_1':
        return Lstm_32_128_10_1()
    if x == 'Lstm_32_128_20_1':
        return Lstm_32_128_20_1()
    if x == 'Lstm_32_256_1_1':
        return Lstm_32_256_1_1()
    if x == 'Lstm_32_256_2_1':
        return Lstm_32_256_2_1()
    if x == 'Lstm_32_256_5_1':
        return Lstm_32_256_5_1()
    if x == 'Lstm_32_256_10_1':
        return Lstm_32_256_10_1()
    if x == 'Lstm_32_256_20_1':
        return Lstm_32_256_20_1()
    if x == 'Lstm_64_16_1_1':
        return Lstm_64_16_1_1()
    if x == 'Lstm_64_16_2_1':
        return Lstm_64_16_2_1()
    if x == 'Lstm_64_16_5_1':
        return Lstm_64_16_5_1()
    if x == 'Lstm_64_16_10_1':
        return Lstm_64_16_10_1()
    if x == 'Lstm_64_16_20_1':
        return Lstm_64_16_20_1()
    if x == 'Lstm_64_32_1_1':
        return Lstm_64_32_1_1()
    if x == 'Lstm_64_32_2_1':
        return Lstm_64_32_2_1()
    if x == 'Lstm_64_32_5_1':
        return Lstm_64_32_5_1()
    if x == 'Lstm_64_32_10_1':
        return Lstm_64_32_10_1()
    if x == 'Lstm_64_32_20_1':
        return Lstm_64_32_20_1()
    if x == 'Lstm_64_64_1_1':
        return Lstm_64_64_1_1()
    if x == 'Lstm_64_64_2_1':
        return Lstm_64_64_2_1()
    if x == 'Lstm_64_64_5_1':
        return Lstm_64_64_5_1()
    if x == 'Lstm_64_64_10_1':
        return Lstm_64_64_10_1()
    if x == 'Lstm_64_64_20_1':
        return Lstm_64_64_20_1()
    if x == 'Lstm_64_128_1_1':
        return Lstm_64_128_1_1()
    if x == 'Lstm_64_128_2_1':
        return Lstm_64_128_2_1()
    if x == 'Lstm_64_128_5_1':
        return Lstm_64_128_5_1()
    if x == 'Lstm_64_128_10_1':
        return Lstm_64_128_10_1()
    if x == 'Lstm_64_128_20_1':
        return Lstm_64_128_20_1()
    if x == 'Lstm_64_256_1_1':
        return Lstm_64_256_1_1()
    if x == 'Lstm_64_256_2_1':
        return Lstm_64_256_2_1()
    if x == 'Lstm_64_256_5_1':
        return Lstm_64_256_5_1()
    if x == 'Lstm_64_256_10_1':
        return Lstm_64_256_10_1()
    if x == 'Lstm_64_256_20_1':
        return Lstm_64_256_20_1()
    if x == 'Lstm_128_16_1_1':
        return Lstm_128_16_1_1()
    if x == 'Lstm_128_16_2_1':
        return Lstm_128_16_2_1()
    if x == 'Lstm_128_16_5_1':
        return Lstm_128_16_5_1()
    if x == 'Lstm_128_16_10_1':
        return Lstm_128_16_10_1()
    if x == 'Lstm_128_16_20_1':
        return Lstm_128_16_20_1()
    if x == 'Lstm_128_32_1_1':
        return Lstm_128_32_1_1()
    if x == 'Lstm_128_32_2_1':
        return Lstm_128_32_2_1()
    if x == 'Lstm_128_32_5_1':
        return Lstm_128_32_5_1()
    if x == 'Lstm_128_32_10_1':
        return Lstm_128_32_10_1()
    if x == 'Lstm_128_32_20_1':
        return Lstm_128_32_20_1()
    if x == 'Lstm_128_64_1_1':
        return Lstm_128_64_1_1()
    if x == 'Lstm_128_64_2_1':
        return Lstm_128_64_2_1()
    if x == 'Lstm_128_64_5_1':
        return Lstm_128_64_5_1()
    if x == 'Lstm_128_64_10_1':
        return Lstm_128_64_10_1()
    if x == 'Lstm_128_64_20_1':
        return Lstm_128_64_20_1()
    if x == 'Lstm_128_128_1_1':
        return Lstm_128_128_1_1()
    if x == 'Lstm_128_128_2_1':
        return Lstm_128_128_2_1()
    if x == 'Lstm_128_128_5_1':
        return Lstm_128_128_5_1()
    if x == 'Lstm_128_128_10_1':
        return Lstm_128_128_10_1()
    if x == 'Lstm_128_128_20_1':
        return Lstm_128_128_20_1()
    if x == 'Lstm_128_256_1_1':
        return Lstm_128_256_1_1()
    if x == 'Lstm_128_256_2_1':
        return Lstm_128_256_2_1()
    if x == 'Lstm_128_256_5_1':
        return Lstm_128_256_5_1()
    if x == 'Lstm_128_256_10_1':
        return Lstm_128_256_10_1()
    if x == 'Lstm_128_256_20_1':
        return Lstm_128_256_20_1()
    if x == 'Lstm_256_16_1_1':
        return Lstm_256_16_1_1()
    if x == 'Lstm_256_16_2_1':
        return Lstm_256_16_2_1()
    if x == 'Lstm_256_16_5_1':
        return Lstm_256_16_5_1()
    if x == 'Lstm_256_16_10_1':
        return Lstm_256_16_10_1()
    if x == 'Lstm_256_16_20_1':
        return Lstm_256_16_20_1()
    if x == 'Lstm_256_32_1_1':
        return Lstm_256_32_1_1()
    if x == 'Lstm_256_32_2_1':
        return Lstm_256_32_2_1()
    if x == 'Lstm_256_32_5_1':
        return Lstm_256_32_5_1()
    if x == 'Lstm_256_32_10_1':
        return Lstm_256_32_10_1()
    if x == 'Lstm_256_32_20_1':
        return Lstm_256_32_20_1()
    if x == 'Lstm_256_64_1_1':
        return Lstm_256_64_1_1()
    if x == 'Lstm_256_64_2_1':
        return Lstm_256_64_2_1()
    if x == 'Lstm_256_64_5_1':
        return Lstm_256_64_5_1()
    if x == 'Lstm_256_64_10_1':
        return Lstm_256_64_10_1()
    if x == 'Lstm_256_64_20_1':
        return Lstm_256_64_20_1()
    if x == 'Lstm_256_128_1_1':
        return Lstm_256_128_1_1()
    if x == 'Lstm_256_128_2_1':
        return Lstm_256_128_2_1()
    if x == 'Lstm_256_128_5_1':
        return Lstm_256_128_5_1()
    if x == 'Lstm_256_128_10_1':
        return Lstm_256_128_10_1()
    if x == 'Lstm_256_128_20_1':
        return Lstm_256_128_20_1()
    if x == 'Lstm_256_256_1_1':
        return Lstm_256_256_1_1()
    if x == 'Lstm_256_256_2_1':
        return Lstm_256_256_2_1()
    if x == 'Lstm_256_256_5_1':
        return Lstm_256_256_5_1()
    if x == 'Lstm_256_256_10_1':
        return Lstm_256_256_10_1()
    if x == 'Lstm_256_256_20_1':
        return Lstm_256_256_20_1()
    if x == 'Lstm_16_16_1_2':
        return Lstm_16_16_1_2()
    if x == 'Lstm_16_16_2_2':
        return Lstm_16_16_2_2()
    if x == 'Lstm_16_16_5_2':
        return Lstm_16_16_5_2()
    if x == 'Lstm_16_16_10_2':
        return Lstm_16_16_10_2()
    if x == 'Lstm_16_16_20_2':
        return Lstm_16_16_20_2()
    if x == 'Lstm_16_32_1_2':
        return Lstm_16_32_1_2()
    if x == 'Lstm_16_32_2_2':
        return Lstm_16_32_2_2()
    if x == 'Lstm_16_32_5_2':
        return Lstm_16_32_5_2()
    if x == 'Lstm_16_32_10_2':
        return Lstm_16_32_10_2()
    if x == 'Lstm_16_32_20_2':
        return Lstm_16_32_20_2()
    if x == 'Lstm_16_64_1_2':
        return Lstm_16_64_1_2()
    if x == 'Lstm_16_64_2_2':
        return Lstm_16_64_2_2()
    if x == 'Lstm_16_64_5_2':
        return Lstm_16_64_5_2()
    if x == 'Lstm_16_64_10_2':
        return Lstm_16_64_10_2()
    if x == 'Lstm_16_64_20_2':
        return Lstm_16_64_20_2()
    if x == 'Lstm_16_128_1_2':
        return Lstm_16_128_1_2()
    if x == 'Lstm_16_128_2_2':
        return Lstm_16_128_2_2()
    if x == 'Lstm_16_128_5_2':
        return Lstm_16_128_5_2()
    if x == 'Lstm_16_128_10_2':
        return Lstm_16_128_10_2()
    if x == 'Lstm_16_128_20_2':
        return Lstm_16_128_20_2()
    if x == 'Lstm_16_256_1_2':
        return Lstm_16_256_1_2()
    if x == 'Lstm_16_256_2_2':
        return Lstm_16_256_2_2()
    if x == 'Lstm_16_256_5_2':
        return Lstm_16_256_5_2()
    if x == 'Lstm_16_256_10_2':
        return Lstm_16_256_10_2()
    if x == 'Lstm_16_256_20_2':
        return Lstm_16_256_20_2()
    if x == 'Lstm_32_16_1_2':
        return Lstm_32_16_1_2()
    if x == 'Lstm_32_16_2_2':
        return Lstm_32_16_2_2()
    if x == 'Lstm_32_16_5_2':
        return Lstm_32_16_5_2()
    if x == 'Lstm_32_16_10_2':
        return Lstm_32_16_10_2()
    if x == 'Lstm_32_16_20_2':
        return Lstm_32_16_20_2()
    if x == 'Lstm_32_32_1_2':
        return Lstm_32_32_1_2()
    if x == 'Lstm_32_32_2_2':
        return Lstm_32_32_2_2()
    if x == 'Lstm_32_32_5_2':
        return Lstm_32_32_5_2()
    if x == 'Lstm_32_32_10_2':
        return Lstm_32_32_10_2()
    if x == 'Lstm_32_32_20_2':
        return Lstm_32_32_20_2()
    if x == 'Lstm_32_64_1_2':
        return Lstm_32_64_1_2()
    if x == 'Lstm_32_64_2_2':
        return Lstm_32_64_2_2()
    if x == 'Lstm_32_64_5_2':
        return Lstm_32_64_5_2()
    if x == 'Lstm_32_64_10_2':
        return Lstm_32_64_10_2()
    if x == 'Lstm_32_64_20_2':
        return Lstm_32_64_20_2()
    if x == 'Lstm_32_128_1_2':
        return Lstm_32_128_1_2()
    if x == 'Lstm_32_128_2_2':
        return Lstm_32_128_2_2()
    if x == 'Lstm_32_128_5_2':
        return Lstm_32_128_5_2()
    if x == 'Lstm_32_128_10_2':
        return Lstm_32_128_10_2()
    if x == 'Lstm_32_128_20_2':
        return Lstm_32_128_20_2()
    if x == 'Lstm_32_256_1_2':
        return Lstm_32_256_1_2()
    if x == 'Lstm_32_256_2_2':
        return Lstm_32_256_2_2()
    if x == 'Lstm_32_256_5_2':
        return Lstm_32_256_5_2()
    if x == 'Lstm_32_256_10_2':
        return Lstm_32_256_10_2()
    if x == 'Lstm_32_256_20_2':
        return Lstm_32_256_20_2()
    if x == 'Lstm_64_16_1_2':
        return Lstm_64_16_1_2()
    if x == 'Lstm_64_16_2_2':
        return Lstm_64_16_2_2()
    if x == 'Lstm_64_16_5_2':
        return Lstm_64_16_5_2()
    if x == 'Lstm_64_16_10_2':
        return Lstm_64_16_10_2()
    if x == 'Lstm_64_16_20_2':
        return Lstm_64_16_20_2()
    if x == 'Lstm_64_32_1_2':
        return Lstm_64_32_1_2()
    if x == 'Lstm_64_32_2_2':
        return Lstm_64_32_2_2()
    if x == 'Lstm_64_32_5_2':
        return Lstm_64_32_5_2()
    if x == 'Lstm_64_32_10_2':
        return Lstm_64_32_10_2()
    if x == 'Lstm_64_32_20_2':
        return Lstm_64_32_20_2()
    if x == 'Lstm_64_64_1_2':
        return Lstm_64_64_1_2()
    if x == 'Lstm_64_64_2_2':
        return Lstm_64_64_2_2()
    if x == 'Lstm_64_64_5_2':
        return Lstm_64_64_5_2()
    if x == 'Lstm_64_64_10_2':
        return Lstm_64_64_10_2()
    if x == 'Lstm_64_64_20_2':
        return Lstm_64_64_20_2()
    if x == 'Lstm_64_128_1_2':
        return Lstm_64_128_1_2()
    if x == 'Lstm_64_128_2_2':
        return Lstm_64_128_2_2()
    if x == 'Lstm_64_128_5_2':
        return Lstm_64_128_5_2()
    if x == 'Lstm_64_128_10_2':
        return Lstm_64_128_10_2()
    if x == 'Lstm_64_128_20_2':
        return Lstm_64_128_20_2()
    if x == 'Lstm_64_256_1_2':
        return Lstm_64_256_1_2()
    if x == 'Lstm_64_256_2_2':
        return Lstm_64_256_2_2()
    if x == 'Lstm_64_256_5_2':
        return Lstm_64_256_5_2()
    if x == 'Lstm_64_256_10_2':
        return Lstm_64_256_10_2()
    if x == 'Lstm_64_256_20_2':
        return Lstm_64_256_20_2()
    if x == 'Lstm_128_16_1_2':
        return Lstm_128_16_1_2()
    if x == 'Lstm_128_16_2_2':
        return Lstm_128_16_2_2()
    if x == 'Lstm_128_16_5_2':
        return Lstm_128_16_5_2()
    if x == 'Lstm_128_16_10_2':
        return Lstm_128_16_10_2()
    if x == 'Lstm_128_16_20_2':
        return Lstm_128_16_20_2()
    if x == 'Lstm_128_32_1_2':
        return Lstm_128_32_1_2()
    if x == 'Lstm_128_32_2_2':
        return Lstm_128_32_2_2()
    if x == 'Lstm_128_32_5_2':
        return Lstm_128_32_5_2()
    if x == 'Lstm_128_32_10_2':
        return Lstm_128_32_10_2()
    if x == 'Lstm_128_32_20_2':
        return Lstm_128_32_20_2()
    if x == 'Lstm_128_64_1_2':
        return Lstm_128_64_1_2()
    if x == 'Lstm_128_64_2_2':
        return Lstm_128_64_2_2()
    if x == 'Lstm_128_64_5_2':
        return Lstm_128_64_5_2()
    if x == 'Lstm_128_64_10_2':
        return Lstm_128_64_10_2()
    if x == 'Lstm_128_64_20_2':
        return Lstm_128_64_20_2()
    if x == 'Lstm_128_128_1_2':
        return Lstm_128_128_1_2()
    if x == 'Lstm_128_128_2_2':
        return Lstm_128_128_2_2()
    if x == 'Lstm_128_128_5_2':
        return Lstm_128_128_5_2()
    if x == 'Lstm_128_128_10_2':
        return Lstm_128_128_10_2()
    if x == 'Lstm_128_128_20_2':
        return Lstm_128_128_20_2()
    if x == 'Lstm_128_256_1_2':
        return Lstm_128_256_1_2()
    if x == 'Lstm_128_256_2_2':
        return Lstm_128_256_2_2()
    if x == 'Lstm_128_256_5_2':
        return Lstm_128_256_5_2()
    if x == 'Lstm_128_256_10_2':
        return Lstm_128_256_10_2()
    if x == 'Lstm_128_256_20_2':
        return Lstm_128_256_20_2()
    if x == 'Lstm_256_16_1_2':
        return Lstm_256_16_1_2()
    if x == 'Lstm_256_16_2_2':
        return Lstm_256_16_2_2()
    if x == 'Lstm_256_16_5_2':
        return Lstm_256_16_5_2()
    if x == 'Lstm_256_16_10_2':
        return Lstm_256_16_10_2()
    if x == 'Lstm_256_16_20_2':
        return Lstm_256_16_20_2()
    if x == 'Lstm_256_32_1_2':
        return Lstm_256_32_1_2()
    if x == 'Lstm_256_32_2_2':
        return Lstm_256_32_2_2()
    if x == 'Lstm_256_32_5_2':
        return Lstm_256_32_5_2()
    if x == 'Lstm_256_32_10_2':
        return Lstm_256_32_10_2()
    if x == 'Lstm_256_32_20_2':
        return Lstm_256_32_20_2()
    if x == 'Lstm_256_64_1_2':
        return Lstm_256_64_1_2()
    if x == 'Lstm_256_64_2_2':
        return Lstm_256_64_2_2()
    if x == 'Lstm_256_64_5_2':
        return Lstm_256_64_5_2()
    if x == 'Lstm_256_64_10_2':
        return Lstm_256_64_10_2()
    if x == 'Lstm_256_64_20_2':
        return Lstm_256_64_20_2()
    if x == 'Lstm_256_128_1_2':
        return Lstm_256_128_1_2()
    if x == 'Lstm_256_128_2_2':
        return Lstm_256_128_2_2()
    if x == 'Lstm_256_128_5_2':
        return Lstm_256_128_5_2()
    if x == 'Lstm_256_128_10_2':
        return Lstm_256_128_10_2()
    if x == 'Lstm_256_128_20_2':
        return Lstm_256_128_20_2()
    if x == 'Lstm_256_256_1_2':
        return Lstm_256_256_1_2()
    if x == 'Lstm_256_256_2_2':
        return Lstm_256_256_2_2()
    if x == 'Lstm_256_256_5_2':
        return Lstm_256_256_5_2()
    if x == 'Lstm_256_256_10_2':
        return Lstm_256_256_10_2()
    if x == 'Lstm_256_256_20_2':
        return Lstm_256_256_20_2()


if __name__ == "__main__":
    main()

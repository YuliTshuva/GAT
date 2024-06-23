from __future__ import division
from __future__ import print_function

import os
import shutil
from os.path import join
import glob
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT

# Constants
CHECKPOINTS = 'checkpoints'
DATASET = "cora"
M = 1e100
OUTPUT_DIR = 'output'
PRESENT_EVERY = 100

# Create a directory for checkpoints
if os.path.exists(join(CHECKPOINTS, DATASET)):
    shutil.rmtree(join(CHECKPOINTS, DATASET))
if os.path.exists(join(OUTPUT_DIR, DATASET)):
    shutil.rmtree(join(OUTPUT_DIR, DATASET))
os.mkdir(join(CHECKPOINTS, DATASET))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
# How much weight to you give to the l2 norm on parameters
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# Number of not improving epochs before stopping
parser.add_argument('--patience', type=int, default=100, help='Patience')

# Read parameters
args = parser.parse_args()

# Consider the computer resources in cuda parameter
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    print("Using cuda resources.")

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1],
                  nhid=args.hidden,
                  nclass=int(labels.max()) + 1,
                  dropout=args.dropout,
                  nheads=args.nb_heads,
                  alpha=args.alpha)

else:
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)

# Set an optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

# Set cuda arguments to
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# Set as variables for pytorch
features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    """
    Training function.
    """
    # Measure time
    t = time.time()

    # Set the model in training mode
    model.train()

    # Zero the gradient of the optimizer
    optimizer.zero_grad()

    # Predict using the model
    output = model(features, adj)  # We want to learn from all graph features

    # Calculate the training set loss and accuracy
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    # Backpropagation
    loss_train.backward()

    # Improve the model with the optimizer
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    # Calculate the validation set loss and accuracy
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if epoch % PRESENT_EVERY == 0:
        print('Epoch: {:04d}'.format(epoch),
              "\n\t",
              'Train Loss: {:.4f}'.format(loss_train.data.item()),
              "\n\t",
              'Train Accuracy: {:.4f}'.format(acc_train.data.item()),
              "\n\t",
              'Validation Loss: {:.4f}'.format(loss_val.data.item()),
              "\n\t",
              'Validation Accuracy: {:.4f}'.format(acc_val.data.item()),
              "\n\t",
              'Time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item(), loss_train.data.item(), acc_val.data.item(), acc_train.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "\n\t",
          "loss: {:.4f}".format(loss_test.data.item()),
          "\n\t",
          "accuracy: {:.4f}".format(acc_test.data.item()))


# Train model
t_total = time.time()
loss_values = []
status_report = []
bad_counter = 0
best = M
best_epoch = 0
for epoch in tqdm(range(args.epochs), desc="Epochs:"):
    # Training epoch and add to loss list
    loss_val, loss_train, acc_val, acc_train = train(epoch)

    # Add values to status report
    status_report += [loss_val, loss_train, acc_val, acc_train]

    # Add validation loss to list
    loss_values.append(loss_val)

    # Save checkpoint
    torch.save(model.state_dict(), join(CHECKPOINTS, DATASET, f'{epoch}.pkl'))

    # Update best epoch's model
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    # If loss did not improved, note that
    else:
        bad_counter += 1

    # Check if exceeded patience
    if bad_counter == args.patience:
        break

    # Remove irrelevant checkpoints
    files = glob.glob(join(CHECKPOINTS, DATASET, '*.pkl'))
    for file in files:
        # Find the epoch number
        epoch_nb = ""
        for char in file[:-4][::-1]:
            if char.isdigit():
                epoch_nb += char
        epoch_nb = int(epoch_nb[::-1])
        # Check if former to best epoch
        if epoch_nb < best_epoch:
            os.remove(file)

# Remove irrelevant checkpoints
files = glob.glob(join(CHECKPOINTS, DATASET, '*.pkl'))
for file in files:
    # Find the epoch number
    epoch_nb = ""
    for char in file[:-4][::-1]:
        if char.isdigit():
            epoch_nb += char
    epoch_nb = int(epoch_nb[::-1])
    # Check if after best epoch
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load(join(CHECKPOINTS, DATASET, '{}.pkl'.format(best_epoch))))

# Testing
compute_test()

### Plotting
# create output directory
save_dir = join(OUTPUT_DIR, DATASET)
os.makedirs(save_dir, exist_ok=True)

# Save the status report
with open(join(save_dir, "status_report.pkl"), "wb") as file:
    pickle.dump(status_report, file)

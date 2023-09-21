"""
Filename: nnet.py

Author: rajiv256

Created on: 08-08-2022

Description: Neural network implementation of a simple 2D network with one
layer and final regression layer.
"""

# begin imports
import os
import sys
import random
import seaborn as sns
import argparse
import logging
import pickle as pkl
import math

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
# end imports

# begin code


def get_args():
    parser = argparse.ArgumentParser("Arguments for a simple neural net impl")
    parser.add_argument("--train_bsz", type=int, default=16)
    parser.add_argument("--val_bsz", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--dataset_name", type=str, default="default")
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()
    return args


class MyDataset(Dataset):

    def __init__(self, dataset):
        super(MyDataset, self).__init__()

        self.x = [item[0] + [1] for item in dataset] # Augmented dim
        
        self.y = [item[1] for item in dataset]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.tensor([self.x[index][i] for i in range(len(self.x[index]))],
                            dtype=torch.float32), torch.tensor(self.y[index],
                                                               dtype=torch.float32)


class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.linear1 = nn.Linear(3, 3)
        self.linearm = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 1)

        self.track = {
            'ep_tr_loss': [],
            'ep_val_loss': [],
            'epoch': [],
            'st_tr_loss': [],
            'step': []
        }

    def _init_weights(self):
        def init(m):
            if type(m) == nn.Linear:
                m.weight.data.fill_(0.1)
                m.bias.data.fill_(0.1)
                # torch.nn.init.xavier_uniform_(m.weight)
                # m.bias.data.fill_(0.1)
        init(self.linear1)
        init(self.linearm)
        init(self.linear2)

    def forward(self, x):
        x = x + self.linear1(x)
        x = x + self.linearm(x)
        x = self.linear2(x)
        return x

    def train_model(self, opt):

        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        if torch.cuda.is_available():
            self.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            self.parameters()), lr=opt.lr)
        criterion = nn.MSELoss(reduction='mean')

        # Clean the track variables
        for k in self.track.keys():
            self.track[k] = []

        self.train()
        self._init_weights()

        # Dataset and dataloader
        dataset_folder = os.path.join(os.getcwd(), 'data', opt.dataset_name)
        train_pkl = os.path.join(dataset_folder, 'train.pkl')
        val_pkl = os.path.join(dataset_folder, 'val.pkl')

        with open(train_pkl, 'rb') as fin:
            train_set = pkl.load(fin)

        with open(train_pkl, 'rb') as fin:
            val_set = pkl.load(fin)
        train_dataset = MyDataset(train_set)
        val_dataset = MyDataset(val_set)
        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=opt.train_bsz, drop_last=True)
        val_loader = DataLoader(val_dataset, shuffle=False,
                                batch_size=opt.val_bsz, drop_last=True)

        step = 0
        pbar = tqdm(range(opt.epochs))

        for epoch in pbar:

            self.train()

            ep_tr_loss = 0.0
            ep_val_loss = 0.0
            tr_batches = 0
            val_batches = 0

            for x, y in train_loader:

                if torch.cuda.is_available():
                    x.cuda()
                    y.cuda()

                optimizer.zero_grad()
                output = self(x)

                loss = torch.sqrt(criterion(output.flatten(), y.flatten()))
                loss.backward()
                optimizer.step()
                self.track['step'].append(step)
                self.track['st_tr_loss'].append(loss.item())
                step += 1

                ep_tr_loss += loss.item()
                tr_batches += 1

            self.eval()
            val_acc = 0.0
            
            yhats = []

            for x, y in val_loader:

                if torch.cuda.is_available():
                    x.cuda()
                    y.cuda()

                output = self(x)
                loss = torch.sqrt(criterion(output.flatten(), y.flatten()))
                
                ep_val_loss += loss.item()
                
                outputs = list(output.flatten().detach().numpy())
                targets = list(y.flatten().detach().numpy())
                correct = 0
                xnumpy = x.detach().numpy()
                for i in range(len(outputs)):
                    xx = xnumpy[i]
                    b = math.ceil(outputs[i])         
                    if outputs[i] >= 0 and targets[i] == 1:
                        correct += 1
                       
                    if outputs[i] < 0 and targets[i] == -1:
                        correct += 1
                    yhats.append([xx[0], xx[1], b])

                    print("out, tgt, cor: ", outputs[i], targets[i], correct)
                # print(outputs, targets, correct)
                val_acc += correct/len(outputs)

                val_batches += 1



            self.track['epoch'].append(epoch)
            self.track['ep_tr_loss'].append(ep_tr_loss/tr_batches)
            self.track['ep_val_loss'].append(ep_val_loss/val_batches)
            print("val acc", val_acc/val_batches)

            pbar.set_description(
                f"epoch:{epoch} | tr: {ep_tr_loss} | val: {ep_val_loss}")
            yhatsplt = sns.scatterplot(x=[item[0] for item in yhats], y=[item[1] for item in yhats], c=[item[2] for item in yhats])
            plt.savefig("resnetyhats.png")

if __name__ == "__main__":

    opt = get_args()
    nnet = NNet()
    nnet.train_model(opt)

    # Plot epoch train and val loss.
    track = nnet.track
    plt.clf()
    sns.lineplot(x=track['epoch'], y=track['ep_tr_loss'])
    sns.lineplot(x=track['epoch'], y=track['ep_val_loss'])
    
    plt.savefig('plot.png')

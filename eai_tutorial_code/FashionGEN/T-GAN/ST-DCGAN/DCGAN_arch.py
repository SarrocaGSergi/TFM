from __future__ import print_function
# %matplotlib inline
import torch
import h5py
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from CondBatchNorm import *


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf, number_classes):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        # input is (nc) x 64 x 64
        self.conv1 = nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False)
        self.cbatchnorm2 = CBN(num_features=self.ndf*2, n_category=number_classes, n_hidden=64)
        # Relu activation layer
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)
        self.cbatchnorm3 = CBN(num_features=self.ndf*4, n_category=number_classes, n_hidden=64)
        # Relu activation layer
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)
        self.cbatchnorm4 = CBN(num_features=self.ndf*8, n_category=number_classes, n_hidden=64)
        # Relu activation layer
        # state size. (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, input, labels):
        # Block 1 (nc) x 64 x 64
        x = self.conv1(input)
        x = self.relu(x)
        # Block 2 (ndf) x 32 x 32
        x = self.conv2(x)
        x = self.cbatchnorm2(x, labels)
        x = self.relu(x)
        # Block 3 (ndf*2) x 16 x 16
        x = self.conv3(x)
        x = self.cbatchnorm3(x, labels)
        x = self.relu(x)
        # Block 4 (ndf*4) x 8 x 8
        x = self.conv4(x)
        x = self.cbatchnorm4(x, labels)
        x = self.relu(x)
        # Block 5 (ndf*8) x 4 x 4
        x = self.conv5(x)
        x = self.sig(x)
        return x


class Generator(nn.Module):
    def __init__(self, ngpu, nz, nc, ngf, number_classes):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.nz = nz
        self.nc = nc
        # input is Z, going into a convolution
        self.convT1 = nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False)
        self.cbatchnorm1 = CBN(num_features=ngf*8, n_category=number_classes, n_hidden=64)
        self.relu = nn.ReLU(True)
        # state size. (ngf*8) x 4 x 4
        self.convT2 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False)
        self.cbatchnorm2 = CBN(num_features=ngf*4, n_category=number_classes, n_hidden=64)
        # Relu activation Layer
        # state size. (ngf*4) x 8 x 8
        self.convT3 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False)
        self.cbatchnorm3 = CBN(num_features=ngf*2, n_category=number_classes, n_hidden=64)
        # Relu activation layer
        # state size. (ngf*2) x 16 x 16
        self.convT4 = nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False)
        self.cbatchnorm4 = CBN(num_features=ngf, n_category=number_classes, n_hidden=256)
        # Relu activation layer
        # state size. (ngf) x 32 x 32
        self.convT5 = nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False)
        self.tan = nn.Tanh()
        # state size. (nc) x 64 x 64

    def forward(self, input, labels):
        # Block1
        x = self.convT1(input)
        x = self.cbatchnorm1(x, labels)
        x = self.relu(x)
        # Block 2 (ngf*8) x 4 x 4
        x = self.convT2(x)
        x = self.cbatchnorm2(x, labels)
        x = self.relu(x)
        # Block 3 (ngf*4) x 8 x 8
        x = self.convT3(x)
        x = self.cbatchnorm3(x, labels)
        x = self.relu(x)
        # Block 4 (ngf*2) x 16 x 16
        x = self.convT4(x)
        x = self.cbatchnorm4(x, labels)
        x = self.relu(x)
        # Block 5 (ngf) x 32 x 32
        x = self.convT5(x)
        x = self.tan(x)
        # state size. (nc) x 64 x 64
        return x

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


def change_label(label, dictionary):
    for i in dictionary:
        if label[3:-2] == dictionary[i]:
            return i


class FashionGen(Dataset):
    def __init__(self, train_root, transform, dictionary, text_transform=True):
        self.transform = transform
        self.text_transform = text_transform
        self.train_root = train_root
        self.dictionary = dictionary
        self.f = h5py.File(train_root, 'r')
        self.images = self.f.get('input_image')
        #self.descriptions = self.f.get('input_description')
        self.categories = self.f.get('input_category')
        self.n_samples = self.images.shape[0]

    def __getitem__(self, idx):
        # Load Dataset
        image = Image.fromarray(self.images[idx])
        #descriptions = self.descriptions[idx]
        categories = self.categories[idx]
        if self.transform and self.text_transform:
            batch = self.transform(image)
            category = f"{categories}"
            #description = f"{descriptions}"
            label = change_label(category, self.dictionary)

            return batch, label

        elif not self.text_transform:
            batch = self.transform(image)
            category = f"{categories}"
            label = category[3:-2]
            return batch, label

    def __len__(self):
        return self.n_samples


class Generator(nn.Module):
    def __init__(self, ngpu, nz, nc, ngf, number_classes, embedding_size):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.nz = nz
        self.nc = nc
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(number_classes, embedding_size)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz+self.embedding_size, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, labels):
        embed = self.embedding(labels).unsqueeze(2).unsqueeze(3)
        input = torch.cat([input, embed], dim=1)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf, number_classes, image_size):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.image_size = image_size
        self.embedding = nn.Embedding(number_classes, image_size*image_size)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc+1, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        embed = self.embedding(labels).view(labels.shape[0], 1, self.image_size, self.image_size)
        input = torch.cat([input, embed], dim=1)
        return self.main(input)


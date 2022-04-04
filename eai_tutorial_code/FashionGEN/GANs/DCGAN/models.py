from __future__ import print_function

import h5py
import torch.nn as nn
from torch.utils.data import Dataset
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
        self.descriptions = self.f.get('input_description')
        self.categories = self.f.get('input_category')
        self.input_productID = self.f.get('input_productID')
        self.index = self.f.get('index')
        self.input_matchedID = self.f.get('input_matchedID')
        self.input_name = self.f.get('input_name')
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


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.n_disc_features = ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.n_disc_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.n_disc_features, self.n_disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.n_disc_features * 2, self.n_disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.n_disc_features * 4, self.n_disc_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.n_disc_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.noise_vector = nz
        self.n_gen_features = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_vector, self.n_gen_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.n_gen_features * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.n_gen_features * 8, self.n_gen_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_gen_features * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.n_gen_features * 4, self.n_gen_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.n_gen_features * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_gen_features),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.n_gen_features, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


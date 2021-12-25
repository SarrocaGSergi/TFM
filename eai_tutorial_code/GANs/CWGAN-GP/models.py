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


class FashionGen(Dataset):
    def __init__(self, train_root, transform):
        self.transform = transform
        self.train_root = train_root
        self.f = h5py.File(train_root, 'r')
        self.images = self.f.get('input_image')
        self.descriptions = self.f.get('input_description')
        self.n_samples = self.images.shape[0]

    def __getitem__(self, idx):
        # Load Dataset
        image = Image.fromarray(self.images[idx])
        descriptions = self.descriptions[idx]
        if self.transform:
            batch = self.transform(image)
            description = f"{descriptions}"

        return batch, description

    def __len__(self):
        return self.n_samples

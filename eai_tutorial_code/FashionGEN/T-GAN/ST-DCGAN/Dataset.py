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
        self.n_samples = self.images.shape[0]

    def __getitem__(self, idx):
        # Load Dataset
        image = Image.fromarray(self.images[idx])
        descriptions = self.descriptions[idx]
        categories = self.categories[idx]
        if self.transform and self.text_transform:
            batch = self.transform(image)
            category = f"{categories}"
            description = f"{descriptions}"
            description2 = description[3:-2]
            label = category[3:-2]

            return batch, description2

        elif not self.text_transform:
            batch = self.transform(image)
            category = f"{categories}"
            label = category[3:-2]
            return batch, label

    def __len__(self):
        return self.n_samples


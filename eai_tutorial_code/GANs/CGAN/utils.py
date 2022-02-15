from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


def select_dataset(dataset_name, data_root, transform, workers=1, batch_size=64):
    if dataset_name == "fashion_mnist":
        dataset = dset.FashionMNIST(root=data_root, train=True, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    elif dataset_name == "mnist":
        dataset = dset.MNIST(root=data_root, train=True, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    elif dataset_name == "cifar10":
        dataset = dset.CIFAR10(root=data_root, train=True, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    elif dataset_name == "celeba":
        dataset = dset.ImageFolder(root=data_root, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    return dataset, dataloader


# Help function to show processed images from tensors
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


#Weight initialization to break symmetry on first backward gradients (Avoid to learn the same on each neuron)
def weights_init(m):
    # custom weights initialization called on netG and netD
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# helper function
def select_n_random(data, labels, n=100):
    #print(len(data))
    print(len(labels))
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


#From labels to conditional-labels
def create_dictionary(labels):
    pass

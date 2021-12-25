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
from models import *
from PIL import Image


def text_processing(labels, model, tokenizer):
    token_labels = tokenizer(labels, padding='max_length', max_length=120, truncation=True, return_tensors='pt')
    outputs = model(**token_labels)
    return outputs


def data_classes(data_name="mnist"):
    # As it is MNIST dataset, images are just ine channel
    if data_name == "fashion_mnist":
        # Number of channels
        nc = 1
        number_classes = 10
        dictionary = {0: "T-Shirt",
                      1: "Trouser",
                      2: "Pullover",
                      3: "Dress",
                      4: "Coat",
                      5: "Sandal",
                      6: "Shirt",
                      7: "Sneaker",
                      8: "Bag",
                      9: "Ankle Boot",
                      }

    elif data_name == "mnist":
        # Number of channels
        nc = 1
        number_classes = 10
        dictionary = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    elif data_name == "cifar10":
        # Number of channels
        nc = 3
        number_classes = 10
        dictionary = {0: "airplane",
                      1: "automobile",
                      2: "bird",
                      3: "cat",
                      4: "deer",
                      5: "dog",
                      6: "frog",
                      7: "horse",
                      8: "ship",
                      9: "truck"
                      }

    elif data_name == "celeba":
        # Number of channels
        nc = 3
        number_classes = 2
        dictionary = {0: "Male",
                      1: "Female"
                      }
    elif data_name == "fashiongen":
        nc = 3
        number_classes = 2
        dictionary = {}

    return nc, number_classes, dictionary

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

    elif dataset_name == "fashiongen":
        dataset = FashionGen(train_root=data_root, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


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


# Weight initialization to break symmetry on first backward gradients (Avoid to learn the same on each neuron)
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


def cond_labels(number_classes):
    x = 0
    fixed_labels = []
    while x < number_classes:
        current_label = np.ones(10)
        current_label = current_label * x
        fixed_labels.append(current_label)
        x += 1

    return fixed_labels


def gradient_penalty(discriminator, labels, real_batch, fake_batch, device='cpu'):
    BATCH_SIZE, C, H, W = real_batch.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = (epsilon*real_batch) + fake_batch*(1-epsilon)

    # Calculate discriminator gradients
    mixed_scores = discriminator(interpolated_images, labels)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_pnlty = torch.mean((gradient_norm-1) ** 2)

    return gradient_pnlty


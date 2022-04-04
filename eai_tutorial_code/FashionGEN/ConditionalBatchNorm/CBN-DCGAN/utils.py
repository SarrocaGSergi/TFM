from __future__ import print_function

# %matplotlib inline
import numpy
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from scipy.interpolate import interp1d
import PIL.Image as Image
from Dataset import *
from DCGAN_arch import *

from FID import *


def get_data_root(database_name="fashiongen"):
    # Defining some variables
    if database_name == "fashion_mnist":
        data_root = "../app/data/fashion_mnist"
        print("Dataset: " + database_name)
    elif database_name == "mnist":
        data_root = "/app/data/mnist"
        print("Dataset: " + database_name)
    elif database_name == "celeba":
        data_root = "/app/data/celeba"
        print("Dataset: " + database_name)
    elif database_name == "fashiongen":
        data_root = "/app/data/fashion_gen/fashion_gen/fashiongen_full_size_train.h5"
        print("Dataset: " + database_name)

    return data_root


def projector_img_labs(dataset, b_size=100, n_workers=1, shuff=True):
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=shuff, num_workers=n_workers)
    real_batch = iter(dataloader)
    images, labels = real_batch.next()
    return images, labels


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
        dictionary = {"T-Shirt": 0,
                      "Trouser": 1,
                      "Pullover": 2,
                      "Dress": 3,
                      "Coat": 4,
                      "Sandal": 5,
                      "Shirt": 6,
                      "Sneaker": 7,
                      "Bag": 8,
                      "Ankle Boot": 9,
                      }

    elif data_name == "mnist":
        # Number of channels
        nc = 1
        number_classes = 10
        dictionary = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
    elif data_name == "celeba":
        # Number of channels
        nc = 3
        number_classes = 2
        dictionary = {0: "Male",
                      1: "Female"
                      }

    elif data_name == "fashiongen":
        nc = 3
        number_classes = 48
        dictionary = {"BACKPACKS": 0, "BAG ACCESSORIES": 1, "BELTS & SUSPENDERS": 2, "BLANKETS": 3,
                      "BOAT SHOES & MOCCASINS": 4, "BOOTS": 5, "BRIEFCASES": 6, "CLUTCHES & POUCHES": 7, "DRESSES": 8,
                      "DUFFLE & TOP HANDLE BAGS": 9, "DUFFLE BAGS": 10, "ESPADRILLES": 11, "EYEWEAR": 12,
                      "FINE JEWELRY": 13, "FLATS": 14, "GLOVES": 15, "HATS": 16, "HEELS": 17, "JACKETS & COATS": 18,
                      "JEANS": 19, "JEWELRY": 20, "JUMPSUITS": 21, "KEYCHAINS": 22, "LACE UPS": 23, "LINGERIE": 24,
                      "LOAFERS": 25, "MESSENGER BAGS": 26, "MESSENGER BAGS & SATCHELS": 27, "MONKSTRAPS": 28,
                      "PANTS": 29, "POCKET SQUARES & TIE BARS": 30, "POUCHES & DOCUMENT HOLDERS": 31, "SANDALS": 32,
                      "SCARVES": 33, "SHIRTS": 34, "SHORTS": 35, "SHOULDER BAGS": 36, "SKIRTS": 37, "SNEAKERS": 38,
                      "SOCKS": 39, "SUITS & BLAZERS": 40, "SWEATERS": 41, "SWIMWEAR": 42, "TIES": 43, "TOPS": 44,
                      "TOTE BAGS": 45, "TRAVEL BAGS": 46, "UNDERWEAR & LOUNGEWEAR": 47}

    return nc, number_classes, dictionary


def change_labels(dictionary, fixed_labels):
    labels = []
    for i in fixed_labels:
        label = dictionary[i]
        labels.append(label)
    return torch.tensor(labels)


def load_dataset(net, dataset_name, data_root, transform, dictionary, workers=1, batch_size=64):
    if dataset_name == "fashion_mnist":
        dataset = dset.FashionMNIST(root=data_root, train=True, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    elif dataset_name == "mnist":
        dataset = dset.MNIST(root=data_root, train=True, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers, drop_last=True)

    elif dataset_name == "celeba":
        dataset = dset.ImageFolder(root=data_root, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    elif dataset_name == "fashiongen":
        if net == "T-DCGAN":
            dataset = FashionGen(train_root=data_root, transform=transform, text_transform=False, dictionary=dictionary)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        else:
            dataset = FashionGen(train_root=data_root, transform=transform, text_transform=True, dictionary=dictionary)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    return dataset, dataloader


def sample_save(sample_grid, save_dir, save_name):
    out_path = os.path.join(save_dir, save_name)
    pil_image = ToPILImage()(sample_grid)
    pil_image.save(out_path)


def load_dataset_sample(dataloader, save_dir, save_name="dataset_sample.png"):
    real_batch = iter(dataloader)
    images, labels = real_batch.next()
    img_grid = torchvision.utils.make_grid(images, normalize=True)
    sample_save(img_grid, save_dir, save_name)
    return img_grid


# Weight initialization to break symmetry on first backward gradients (Avoid to learn the same on each neuron)
def weights_init(m):
    # custom weights initialization called on netG and netD
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)




# helper function
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def cond_labels(number_classes):
    x = 0
    fixed_labels = []
    while x < number_classes:
        current_label = np.ones(2)
        current_label = current_label * x
        fixed_labels.append(current_label)
        x += 1

    return np.array(fixed_labels)


def load_FID():
    # Preparing Evaluation Metric
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model = model.cuda()
    return model


def gradient_penalty(discriminator, real_batch, fake_batch, l, labels=True, device='cpu'):
    BATCH_SIZE, C, H, W = real_batch.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = (epsilon * real_batch) + fake_batch * (1 - epsilon)

    if labels:
        # Calculate discriminator gradients
        mixed_scores = discriminator(interpolated_images, l)
    else:
        # Calculate discriminator gradients
        mixed_scores = discriminator(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_pnlty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_pnlty


def disc_versus_gen(g_loss, d_loss, out_path):
    plt.figure(figsize=(150, 75))
    plt.title("Generator v.s Discriminator")
    plt.plot(g_loss, label='G', linewidth=10)
    plt.plot(d_loss, label='D', linewidth=10)
    plt.xlabel("Iterations", fontsize=200)
    plt.ylabel("Loss", fontsize=200)
    plt.yticks(fontsize=200)
    plt.xticks(fontsize=200)
    plt.legend(fontsize=175)
    path = out_path + "/" + "G-vs-D.png"
    plt.savefig(path)


def plot_loss(graph, name, out_path, color='b'):
    save_name = name + '.png'
    inside_name = name.replace("_", " ")
    plt.figure(figsize=(150, 75))
    plt.title(f"{inside_name} Progression", fontsize=200)
    plt.plot(graph, label=f"{inside_name}", color=color, linewidth=10)
    if name == "FID":
        plt.xlabel("Epochs", fontsize=200)
    else:
        plt.xlabel("Iterations", fontsize=200)
    plt.ylabel("Loss", fontsize=200)
    plt.yticks(fontsize=200)
    plt.xticks(fontsize=200)
    plt.legend(fontsize=175)
    path = out_path + f"/{save_name}"
    plt.savefig(path)


def smooth_graph(graph, name, out_path, smooth_param=0.9, color='b'):
    save_name = name + '.png'
    inside_name = name.replace("_", " ")
    alpha = len(graph) * (1 - smooth_param)
    x = np.linspace(0, len(graph), len(graph), endpoint=True)
    f = interp1d(x, graph, kind='linear')
    new_graph = np.linspace(0, alpha, len(graph), endpoint=True)
    new_graph = f(new_graph)
    plt.figure(figsize=(150, 75))
    plt.title(f"{inside_name} Progression", fontsize=200)
    plt.plot(new_graph, label=f"{inside_name}", color=color, linewidth=10)
    plt.xlabel("Iterations", fontsize=200)
    plt.ylabel("Loss", fontsize=200)
    plt.yticks(fontsize=200)
    plt.xticks(fontsize=200)
    plt.legend(fontsize=175)
    path = out_path + f"/{save_name}"
    plt.savefig(path)


def create_gif(grid, save_dir):
    save_dir = os.path.join(save_dir, 'noise_progression.gif')
    l = []
    for img in grid:
        pil_image = ToPILImage()(img)
        l.append(pil_image)

    l[0].save(save_dir, save_all=True, append_images=l[1:],
              optimize=False, duration=500, loop=0)


def fixed_one_hot(data):
    ts = []
    for label in data:
        zero_hot = np.zeros(len(data), dtype=int)
        zero_hot[label] = 1
        ts.append(zero_hot)
    ts = np.array(ts)
    tensor = torch.FloatTensor(ts)
    return tensor


def training_one_hot(data,dictionary):
    ts = []
    for label in data:
        zero_hot = np.zeros(len(dictionary), dtype=int)
        zero_hot[label] = 1
        ts.append(zero_hot)
    ts = np.array(ts)
    tensor = torch.FloatTensor(ts)
    return tensor



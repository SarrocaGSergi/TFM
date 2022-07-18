from __future__ import print_function

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from scipy.interpolate import interp1d
import PIL.Image as Image
from models import *
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
        dictionary = {0: "BACKPACKS",
                      1: "BAG ACCESSORIES",
                      2: "BELTS & SUSPENDERS",
                      3: "BLANKETS",
                      4: "BOAT SHOES & MOCCASINS",
                      5: "BOOTS",
                      6: "BRIEFCASES",
                      7: "CLUTCHES & POUCHES",
                      8: "DRESSES",
                      9: "DUFFLE & TOP HANDLE BAGS",
                      10: "DUFFLE BAGS",
                      11: "ESPADRILLES",
                      12: "EYEWEAR",
                      13: "FINE JEWELRY",
                      14: "FLATS",
                      15: "GLOVES",
                      16: "HATS",
                      17: "HEELS",
                      18: "JACKETS & COATS",
                      19: "JEANS",
                      20: "JEWELRY",
                      21: "JUMPSUITS",
                      22: "KEYCHAINS",
                      23: "LACE UPS",
                      24: "LINGERIE",
                      25: "LOAFERS",
                      26: "MESSENGER BAGS",
                      27: "MESSENGER BAGS & SATCHELS",
                      28: "MONKSTRAPS",
                      29: "PANTS",
                      30: "POCKET SQUARES & TIE BARS",
                      31: "POUCHES & DOCUMENT HOLDERS",
                      32: "SANDALS",
                      33: "SCARVES",
                      34: "SHIRTS",
                      35: "SHORTS",
                      36: "SHOULDER BAGS",
                      37: "SKIRTS",
                      38: "SNEAKERS",
                      39: "SOCKS",
                      40: "SUITS & BLAZERS",
                      41: "SWEATERS",
                      42: "SWIMWEAR",
                      43: "TIES",
                      44: "TOPS",
                      45: "TOTE BAGS",
                      46: "TRAVEL BAGS",
                      47: "UNDERWEAR & LOUNGEWEAR"}

    return nc, number_classes, dictionary


def change_label(labels, dictionary):
    y = []
    for label in labels:
        for i in dictionary:
            if label == dictionary[i]:
                y.append(i)
    return torch.LongTensor(np.array(y))


def load_dataset(net, dataset_name, data_root, transform, dictionary, workers=1, batch_size=64):

    if dataset_name == "fashion_mnist":
        dataset = dset.FashionMNIST(root=data_root, train=True, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    elif dataset_name == "mnist":
        dataset = dset.MNIST(root=data_root, train=True, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    elif dataset_name == "celeba":
        dataset = dset.ImageFolder(root=data_root, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    elif dataset_name == "fashiongen":
        if net == "T-WGAN":
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
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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

    return fixed_labels


def load_FID():
    # Preparing Evaluation Metric
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model = model.cuda()
    return model


def gradient_penalty(discriminator, real_batch, fake_batch, l, labels=True, device='cpu'):
    BATCH_SIZE, C, H, W = real_batch.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = (epsilon*real_batch) + fake_batch*(1-epsilon)

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
    gradient_pnlty = torch.mean((gradient_norm-1) ** 2)

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
    alpha = len(graph) * (1-smooth_param)
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

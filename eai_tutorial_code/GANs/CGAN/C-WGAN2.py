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
from utils import *

torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
dataset_name = "cifar10"
opt = "Adam"
net = "CWGAN"


# Defining some variables
if dataset_name == "fashion_mnist":
    data_root = "../app/data/fashion_mnist"
    print("Dataset: " + dataset_name)
elif dataset_name == "mnist":
    data_root = "../app/data/mnist"
    print("Dataset: " + dataset_name)
elif dataset_name == "cifar10":
    data_root = "/app/data/cifar10"
    print("Dataset: " + dataset_name)

# number of workers for dataloaders
workers = 2
batch_size = [64]
image_size = 64
# Number of channels
# As it is MNIST dataset, images are just ine channel
if dataset_name == "fashion_mnist" or dataset_name == "mnist":
    nc = 1
    number_classes = 10
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
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

elif dataset_name == "cifar10":
    nc = 3
    number_classes = 10
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
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

# size of z latent vector
nz = 100
embedding_size = 100
# size of feature maps in discriminator
ndf = 64

# size of feature maps in generator
ngf = 64

num_epochs = 100

learning_rates = [2e-4]

# hyperparam for Adam optimizers
beta1 = 0.5

# number of gpus available
ngpu = 1

# Critic Iterations
critic_iter = 5

# Weight Clip

weight_clip = 0.01

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                ])#transforms.Normalize((0.5, 0.5, 0.5),
                                                     #(0.5, 0.5, 0.5))

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu, number_classes, embedding_size):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.embedding = torch.nn.Embedding(number_classes, embedding_size)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + embedding_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, labels):
        embed = self.embedding(labels).unsqueeze(2).unsqueeze(3)
        input = torch.cat([input, embed], dim=1)
        return self.main(input)


# Create the generator
netG = Generator(ngpu, number_classes, embedding_size).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu, number_classes, image_size):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.embedding = torch.nn.Embedding(number_classes, image_size * image_size)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input, labels):
        embed = self.embedding(labels).view(labels.shape[0], 1, image_size, image_size)
        input = torch.cat([input, embed], dim=1)
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu, number_classes, image_size).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(number_classes, nz, 1, 1, device=device)
fixed_labels = Variable(torch.LongTensor(np.arange(number_classes))).cuda()
# Establish convention for real and fake labels during training

real_label = 1.
fake_label = 0.

'''
    # select random images and their target indices
    #images, labels = select_n_random(dataset.data, dataset.targets)
    imgs = dataset.data
    labels = dataset.targets

    # get the class labels for each image
    if dataset_name == 'fashion_mnist':
        class_labels = [dictionary[lab] for lab in labels.numpy()]
    elif dataset_name == 'mnist':
        class_labels = [dictionary[lab] for lab in labels]
    elif dataset_name == 'cifar10':
        class_labels = [dictionary[lab] for lab in labels]

    # log embeddings
    features = imgs.reshape(50000, 3, 32, 32)
    writer.add_embedding(features,
                         metadata=class_labels,
                         label_img=imgs.unsqueeze(2))
    writer.close()'''




for bs in batch_size:
    dataset, dataloader = select_dataset(dataset_name=dataset_name, data_root=data_root, transform=transform,
                                         workers=workers, batch_size=bs)


    for lr in learning_rates:

        # Training Loop
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        # Setup RMSprop optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.9))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.9))

        print("Creating summary writer...")
        directory = "/app/cgan_results/"
        print(directory)
        print("Creating Summary at: " + directory)
        writer = SummaryWriter(directory + "HpO" + "/" + net + "/" + opt + "/" + f"{bs}" + "/" + f"{lr}")
        print("Summary Created on: " + directory + "HpO" + "/" + net + "/" + opt + "/" + f"{bs}" + "/" + f"{lr}")

        # Showing a batch of training images in tensorboard
        real_batch = iter(dataloader)
        images, labels = real_batch.next()
        img_grid = torchvision.utils.make_grid(images)
        matplotlib_imshow(img_grid, one_channel=False)
        writer.add_image('Training images', img_grid)

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):

                for _ in range(critic_iter):

                    ## Train with all-real batch
                    netD.zero_grad()

                    # Format batch
                    real_cpu = data[0].to(device)
                    y = data[1].to(device)
                    b_size = real_cpu.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                    label2 = torch.full((b_size,), real_label, dtype=torch.float, device=device)

                    # Forward pass real batch through D
                    output1 = netD(real_cpu, y).view(-1)

                    ## Train with all-fake batch
                    # Generate batch of latent vectors
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    # Generate fake image batch with G
                    fake = netG(noise, y)
                    label2.fill_(fake_label)
                    # Classify all fake batch with D
                    output2 = netD(fake.detach(), y).view(-1)
                    D_x = output1.mean().item()
                    D_G_z1 = output2.mean().item()
                    # WLoss function
                    loss = -(torch.mean(output1) - torch.mean(output2))

                    loss.backward(retain_graph=True)

                    # Update D
                    optimizerD.step()
                    D_losses.append(loss.item())

                    for p in netD.parameters():
                        p.data.clamp_(-weight_clip, weight_clip)

                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake, y).view(-1)
                # Calculate G's loss based on this output
                loss_g = -torch.mean(output)
                # Calculate gradients for G
                loss_g.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             loss.item(), loss_g.item(), D_x, D_G_z1, D_G_z2))
                    ################
                    # Discriminator
                    ################
                    writer.add_scalar('Discriminator Loss', loss, global_step=iters)
                    ################
                    # Generator
                    ################
                    writer.add_scalar('Generator Loss', loss_g, global_step=iters)

                # Save Losses for plotting later
                G_losses.append(loss_g.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0):
                    with torch.no_grad():
                        fake = netG(fixed_noise, fixed_labels).detach().cpu()

                        fake_grid = torchvision.utils.make_grid(fake)
                        matplotlib_imshow(fake_grid, one_channel=False)
                        writer.add_image('Generated Images', fake_grid, global_step=iters)
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
        writer.add_hparams({"lr": lr, "bsize": bs}, {"DLoss": sum(D_losses)/len(D_losses), "GLoss": sum(G_losses)/len(G_losses)})
        writer.close()

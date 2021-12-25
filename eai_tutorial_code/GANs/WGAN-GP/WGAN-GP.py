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
dataset_name = "celeba"
opt = "Adam_transformed"
net = "WGAN-GP"


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
elif dataset_name == "celeba":
    data_root = "/app/data/celeba"
    print("Dataset: " + dataset_name)
# Hyper-parameters selection

# number of workers for dataloaders
workers = 2
batch_size = 64
image_size = 64

nc, number_classes, dictionary = data_classes(data_name=dataset_name)

# size of z latent vector
nz = 100
# size of feature maps in discriminator
ndf = 64

# size of feature maps in generator
ngf = 64

num_epochs = 15

learning_rates = 2e-4

# hyperparam for Adam optimizers
beta1 = 0.5
beta2 = 0.9

# number of gpus available
ngpu = 1

# Critic Iterations
critic_iter = 5

# Lambda operator
lmda = 10


transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5 for _ in range(nc)], [0.5 for _ in range(nc)]),
                                ])


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
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

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d((ndf * 2), affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d((ndf * 4), affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d((ndf * 8), affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)
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
fixed_noise = torch.randn(100, nz, 1, 1, device=device)
# Establish convention for real and fake labels during training

real_label = 1.
fake_label = 0.

dataset, dataloader = select_dataset(dataset_name=dataset_name, data_root=data_root, transform=transform,
                                         workers=workers, batch_size=batch_size)

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
generated_list = []
iters = 0

# Setup optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=learning_rates, betas=(beta1, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rates, betas=(beta1, 0.9))

# Plotting on Tensorboard
print("Creating summary writer...")
directory = "/app/gan_results/"
print("Creating Summary at: " + directory)
writer = SummaryWriter(directory + net + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{learning_rates}")
print("Summary Created on: " + directory + net + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{learning_rates}")

# Showing a batch of training images in tensorboard
real_batch = iter(dataloader)
images, labels = real_batch.next()
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=False)
writer.add_image('Training images', img_grid)

# Training Loop
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

            # Forward pass real batch through D
            real_critic = netD(real_cpu).view(-1)

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)

            # Generate fake image batch with G
            fake = netG(noise)

            # Classify all fake batch with D
            fake_critic = netD(fake.detach()).view(-1)
            D_x = real_critic.mean().item()
            D_G_z1 = fake_critic.mean().item()

            # WLoss function + WGP
            gp = gradient_penalty(discriminator=netD, real_batch=real_cpu, fake_batch=fake, device=device)
            loss = -(torch.mean(real_critic) - torch.mean(fake_critic)) + lmda*gp

            # Back-propagate
            loss.backward(retain_graph=True)

            # Update D
            optimizerD.step()
            D_losses.append(loss.item())

        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)

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
                fake = netG(fixed_noise).detach().cpu()
                fake_grid = torchvision.utils.make_grid(fake, normalize=True)
                writer.add_image('Generated Images', fake_grid, global_step=iters)

        iters += 1

writer.close()

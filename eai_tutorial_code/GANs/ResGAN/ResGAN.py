from __future__ import print_function
# %matplotlib inline
import argparse
import os
import math
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.nn.utils.spectral_norm import spectral_norm
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
net = "ResGAN"

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
batch_size = [64]
image_size = 64

number_classes, dictionary, nc = data_classes(data_name=dataset_name)

# size of z latent vector
nz = 128
embedding_size = 100
# size of feature maps in discriminator
ndf = 64

# size of feature maps in generator
ngf = 64

num_epochs = 10

learning_rates = [2e-4]

# hyperparam for Adam optimizers
beta1 = 0.0
beta2 = 0.9

# number of gpus available
ngpu = 1

# Critic Iterations
critic_iter = 5

# Lambda operator
lmda = 10

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5 for _ in range(nc)], [0.5 for _ in range(nc)]),
                                ])

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)


def res_arch_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if 'residual' in name:
                init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)



class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResGenerator32(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 4 * 4 * 256)

        self.blocks = nn.Sequential(
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        res_arch_init(self)

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 256, 4, 4)
        return self.blocks(inputs)


# Create the generator
netG = ResGenerator32(nz).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Print the model
print(netG)


class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        self.residual = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
            nn.AvgPool2d(2))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x))


class ResDiscriminator32(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 128),
            ResDisBlock(128, 128, down=True),
            ResDisBlock(128, 128),
            ResDisBlock(128, 128),
            nn.ReLU())
        self.linear = spectral_norm(nn.Linear(128, 1, bias=False))
        res_arch_init(self)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x




# Create the Discriminator
netD = ResDiscriminator32().to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Print the model
print(netD)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
f_labels = cond_labels(number_classes=number_classes)
fixed_noise = torch.randn(100, nz, device=device)
fixed_labels = Variable(torch.LongTensor(f_labels)).view(-1).cuda()
# Establish convention for real and fake labels during training

real_label = 1.
fake_label = 0.


for bs in batch_size:
    dataset, dataloader = select_dataset(dataset_name=dataset_name, data_root=data_root, transform=transform,
                                         workers=workers, batch_size=bs)

    for lr in learning_rates:
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        generated_list = []
        iters = 0

        # Setup optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.9))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.9))

        # Plotting on Tensorboard
        print("Creating summary writer...")
        directory = "/app/gan_results/"
        print("Creating Summary at: " + directory)
        writer = SummaryWriter(
            directory + net + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{bs}" + "/" + f"{lr}")
        print(
            "Summary Created on: " + directory + net + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{bs}" + "/" + f"{lr}")

        # Showing a batch of training images in tensorboard
        real_batch = iter(dataloader)
        images, labels = real_batch.next()
        img_grid = torchvision.utils.make_grid(images, normalize=True)
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
                    real_gpu = data[0].to(device)
                    y = data[1].to(device)
                    b_size = real_gpu.size(0)

                    # Forward pass real batch through D
                    real_critic = netD(real_gpu).view(-1)


                    ## Train with all-fake batch
                    # Generate batch of latent vectors
                    noise = torch.randn(b_size, nz, device=device)
                    # Generate fake image batch with G
                    fake = netG(noise)


                    # Classify all fake batch with D
                    fake_critic = netD(fake.detach()).view(-1)

                    D_x = real_critic.mean().item()
                    D_G_z1 = fake_critic.mean().item()

                    # WLoss function + WGP
                    gp = gradient_penalty(discriminator=netD, real_batch=real_gpu, fake_batch=fake,
                                          device=device)
                    loss = -(torch.mean(real_critic) - torch.mean(fake_critic)) + lmda * gp

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
                if iters % 500 == 0:
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                        fake_grid = torchvision.utils.make_grid(fake, normalize=True)
                        writer.add_image('Generated Images', fake_grid, global_step=iters)

                iters += 1
        writer.close()

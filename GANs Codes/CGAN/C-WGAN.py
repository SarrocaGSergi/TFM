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

torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Defining some variables

data_root = "../app/data/mnist"
# number of workers for dataloaders
workers = 2
batch_size = 128
image_size = 64
# Number of channels
# As it is MNIST dataset, images are just ine channel
nc = 1
number_classes = 10
# size of z latent vector
nz = 100
embedding_size = 100
# size of feature maps in discriminator
ndf = 64

# size of feature maps in generator
ngf = 64

num_epochs = 5
lr = 5e-5

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
                                transforms.Normalize((0.5), (0.5))])
dataset = dset.MNIST(root=data_root, train=True, transform=transform, download=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


print("Creating summary writer...")

directory = "../app/cgan_results/c-wgan_results"
writer = SummaryWriter(directory)

print("Summary Created on: " + directory)

real_batch = iter(dataloader)
images, labels = real_batch.next()
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
writer.add_image('Training images', img_grid)


# custom weights initialization called on netG and netD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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
fixed_noise = torch.randn(10, nz, 1, 1, device=device)
fixed_labels = Variable(torch.LongTensor(np.arange(10))).cuda()
# Establish convention for real and fake labels during training

real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


# helper function
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


# select random images and their target indices
images, labels = select_n_random(dataset.data, dataset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                     metadata=class_labels,
                     label_img=images.unsqueeze(1))
writer.close()

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

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
            # Loss function
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
            # Discriminator
            ################
            writer.add_scalar('Generator Loss', loss_g, global_step=iters)
            fake_grid = torchvision.utils.make_grid(netG(fixed_noise, fixed_labels).detach().cpu())
            matplotlib_imshow(fake_grid, one_channel=True)
            writer.add_image('Generated Images', fake_grid)

        # Save Losses for plotting later
        G_losses.append(loss_g.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_labels).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

writer.close()

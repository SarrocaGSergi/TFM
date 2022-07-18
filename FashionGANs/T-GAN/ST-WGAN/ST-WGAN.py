from __future__ import print_function

import os
import torch
import torchvision
import argparse
import random
from torch import optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer

from WGAN_Arch import *
from utils import *

torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 2

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(f"DEVICE: {device}")

# Defining Dataset, Optimizer & Architecture
dataset_name = "fashiongen"
opt = "Adam"
network = "ST-WGAN"

# Defining some variables
TRAIN_ROOT = get_data_root(database_name=dataset_name, train=True)
VAL_ROOT = get_data_root(database_name=dataset_name, train=False)

# Number of workers for dataloader
n_workers = 4

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transform.
image_size = 64

nc, number_classes, dictionary = data_classes(data_name=dataset_name)

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 20
embedding_size = 768

# Critic Iterations
critic_iter = 5

# Weight Clip
weight_clip = 0.01

learning_rate = 2e-4
print(f'Learning rate: {learning_rate}')

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
beta2 = 0.99

TRANSFORM = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5 for _ in range(nc)], [0.5 for _ in range(nc)]),
                                ])

print("Loading Data...")

train_dataset, train_dataloader = load_dataset(net=network, dataset_name=dataset_name, train=True, data_root=TRAIN_ROOT, transform=TRANSFORM,
                                         dictionary=dictionary, workers=n_workers, batch_size=batch_size)

val_dataset, val_dataloader = load_dataset(net=network, dataset_name=dataset_name, train=False, data_root=VAL_ROOT, transform=TRANSFORM,
                                         dictionary=dictionary, workers=n_workers, batch_size=batch_size)

print("Data loaded.")

# Load Architecture
# Create the Discriminator
netD = WGAN_Discriminator(ngpu, nc, ndf, embedding_size).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
netD.apply(weights_init)
# Print the model
print(netD)
# Generator Code
# Create the generator
netG = WGAN_Generator(ngpu, nz, nc, ngf, embedding_size).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
# Print the model
print(netG)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/distilbert-base-nli-mean-tokens").to(device)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator

_, FIXED_DESCRIPTIONS = get_fixed_descriptions(dataset=val_dataset)

FIXED_NOISE = torch.randn(64, nz, 1, 1, device=device)

# Setup Adam optimizers for both G and D
# optimizerD = optim.RMSprop(netD.parameters(), lr=learning_rate)
# optimizerG = optim.RMSprop(netG.parameters(), lr=learning_rate)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, beta2))


# Plotting on Tensorboard
print("Creating summary writer...")
directory = "/app/final_results/T-GANs/"
OUT_PATH = directory + network + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{learning_rate}"
GEN_WEIGHTS = directory + network + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{learning_rate}"
DISC_WEIGHTS = directory + network + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{learning_rate}"
writer = SummaryWriter(OUT_PATH)
print("Summary Created on: " + OUT_PATH)

f = open(os.path.join(OUT_PATH, "fixed_sequences.txt"), 'w')
[f.write((sentence + ' \n')) for sentence in FIXED_DESCRIPTIONS]
f.close()

SAMPLE_IMAGE = load_dataset_sample(dataloader=train_dataloader, save_dir=OUT_PATH)
writer.add_image('Training images', SAMPLE_IMAGE)

# Training Loop
# Lists to keep track of progress
IMG_LIST = []
G_losses = []
D_losses = []
FID_SCORES = []
ITERS = 0
FID_MODEL = load_FID()
# FIXED_IMAGES, TARGETS = projector_img_labs(dataset=dataset, b_size=100, n_workers=2)

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(train_dataloader, 0):

        for _ in range(critic_iter):

            ## Train with all-real batch
            netD.zero_grad()

            # Format batch
            real_cpu = data[0].to(device)
            bat_size = real_cpu.size(0)
            y = list(data[1])
            encoded_tokens = tokenizer(y, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = model(**encoded_tokens)
            sequence_embeddings = mean_pooling(model_output, encoded_tokens['attention_mask']).to(device)

            # Forward pass real batch through D
            output1 = netD(real_cpu, sequence_embeddings).view(-1)

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(bat_size, nz, 1, 1, device=device)

            # Generate fake image batch with G
            fake = netG(noise, sequence_embeddings)

            # Classify all fake batch with D
            output2 = netD(fake.detach(), sequence_embeddings).view(-1)
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
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake, sequence_embeddings).view(-1)
        # Calculate G's loss based on this output
        loss_g = -torch.mean(output)
        # Calculate gradients for G
        loss_g.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Save G Losses for plotting later
        G_losses.append(loss_g.item())

        # Output training stats
        if i % 250 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(train_dataloader),
                     loss.item(), loss_g.item()))
            ################
            # Discriminator
            ################
            writer.add_scalar('Discriminator Loss', loss, global_step=ITERS)
            ################
            # Generator
            ################
            writer.add_scalar('Generator Loss', loss_g, global_step=ITERS)
            ################
            # Projector
            ################
            '''features = FIXED_IMAGES.reshape(FIXED_IMAGES.shape[0], -1)
            class_labels = [dictionary[int(lab)] for lab in TARGETS]
            writer.add_embedding(features, metadata=class_labels, label_img=FIXED_IMAGES, global_step=ITERS)'''

        # Check how the generator is doing by saving G's output on fixed_noise
        if (ITERS % 250 == 0) or ((epoch == num_epochs - 1) and (i == len(train_dataloader) - 1)):
            with torch.no_grad():
                FIXED_ENCODED = tokenizer(list(FIXED_DESCRIPTIONS), padding=True, truncation=True, return_tensors='pt').to(device)
                with torch.no_grad():
                    FIXED_M_OUT = model(**FIXED_ENCODED)
                FIXED_EMBEDDINGS = mean_pooling(FIXED_M_OUT, FIXED_ENCODED['attention_mask']).to(device)
                fake2 = netG(FIXED_NOISE, FIXED_EMBEDDINGS).detach().cpu()
                fake_grid = torchvision.utils.make_grid(fake2, padding=2, normalize=True)
                writer.add_image('Generated Images', fake_grid, global_step=ITERS)
                IMG_LIST.append(fake_grid)
        ITERS += 1

    FID = calculate_fretchet(real_cpu, fake, FID_MODEL)
    FID_SCORES.append(FID)
    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFID: %0.4f'
          % (epoch, num_epochs, i, len(train_dataloader),
             loss.item(), loss_g.item(), FID))
    writer.add_scalar('FID', FID, global_step=ITERS)

    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': netD.state_dict(),
            'loss': loss,
        }, DISC_WEIGHTS + f"/netD_checkpoint_{epoch}.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': netG.state_dict(),
            'loss': loss_g,
        }, DISC_WEIGHTS + f"/netG_checkpoint_{epoch}.pt")

        torch.save(netD, GEN_WEIGHTS + f"/netD_full_{epoch}.pt")
        torch.save(netG, GEN_WEIGHTS + f"/netG_full_{epoch}.pt")

writer.close()

# Statistics Plot
print('Plotting results...')
plot_loss(graph=D_losses, name="Discriminator", out_path=OUT_PATH, color='r')
smooth_graph(graph=D_losses, name="Discriminator_Smooth", out_path=OUT_PATH)
plot_loss(graph=G_losses, name="Generator", out_path=OUT_PATH)
smooth_graph(graph=G_losses, name="Generator_Smooth", out_path=OUT_PATH, color='y')
plot_loss(graph=FID_SCORES, name="FID", out_path=OUT_PATH)
disc_versus_gen(g_loss=G_losses, d_loss=D_losses, out_path=OUT_PATH)
sample_save(fake_grid, save_dir=OUT_PATH, save_name=f"{network}_gen_images.png")
create_gif(grid=IMG_LIST, save_dir=OUT_PATH)


print('Plotted: \n'
      '-  Discriminator graph progression. \n'
      '-  Smoothed Discriminator graph progression. \n'
      '-  Generator graph progression. \n'
      '-  Smoothed Generator graph progression. \n'
      '-  FID graph progression. \n'
      '-  Last image generated sample. \n'
      '-  Gif with the image generation progression.')

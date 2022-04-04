from __future__ import print_function

import torchvision
import random
from torch import optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from utils import *

torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# Defining Dataset, Optimizer & Architecture
dataset_name = "fashiongen"
opt = "Adam"
network = "CGAN"

# Defining some variables
DATA_ROOT = get_data_root(database_name=dataset_name)

# Number of workers for dataloader
n_workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transform.
image_size = 64

nc, number_classes, dictionary = data_classes(data_name=dataset_name)

# Size of z latent vector (i.e. size of generator input)
nz = 100
embedding_size = 100
# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 20

# Learning rate for optimizers
learning_rate = 2e-4

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

dataset, dataloader = load_dataset(net=network, dataset_name=dataset_name, data_root=DATA_ROOT, transform=TRANSFORM,
                                   dictionary=dictionary, workers=n_workers, batch_size=batch_size)

print("Data loaded.")

# Generator Code
# Create the generator
netG = Generator(ngpu, nz, nc, ngf, number_classes, embedding_size).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


# Create the Discriminator
netD = Discriminator(ngpu, nc, ndf, number_classes, image_size).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
F_LABELS = cond_labels(number_classes=number_classes)
FIXED_NOISE = torch.randn(96, nz, 1, 1, device=device)
FIXED_LABELS = Variable(torch.LongTensor(F_LABELS)).view(-1).cuda()


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, beta2))

# Plotting on Tensorboard
print("Creating summary writer...")
directory = "/app/gan_results/final_results/Conditionals/"
OUT_PATH = directory + network + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{learning_rate}"
GEN_WEIGHTS = directory + network + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{learning_rate}+/gen_weights"
DISC_WEIGHTS = directory + network + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{learning_rate}+/disc_weights"
writer = SummaryWriter(OUT_PATH)
print("Summary Created on: " + OUT_PATH)

SAMPLE_IMAGE = load_dataset_sample(dataloader=dataloader, save_dir=OUT_PATH)
writer.add_image('Training images', SAMPLE_IMAGE)

# Training Loop
# Lists to keep track of progress
IMG_LIST = []
G_losses = []
D_losses = []
FID_SCORES = []
ITERS = 0
FID_MODEL = load_FID()
FIXED_IMAGES, TARGETS = projector_img_labs(dataset=dataset, b_size=200, n_workers=2)

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        bat_size = real_cpu.size(0)
        y = data[1].to(device)
        labels = torch.full((bat_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu, y).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, labels)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(bat_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise, y)
        labels.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach(), y).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, labels)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake, y).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, labels)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Output training stats
        if i % 250 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item()))
            ################
            # Discriminator
            ################
            writer.add_scalar('Discriminator Loss', errD, global_step=ITERS)
            ################
            # Generator
            ################
            writer.add_scalar('Generator Loss', errG, global_step=ITERS)
            ################
            # Projector
            ################
            features = FIXED_IMAGES.reshape(FIXED_IMAGES.shape[0], -1)
            class_labels = [dictionary[int(lab)] for lab in TARGETS]
            writer.add_embedding(features, metadata=class_labels, label_img=FIXED_IMAGES, global_step=ITERS)
        # Check how the generator is doing by saving G's output on fixed_noise
        if (ITERS % 250 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake2 = netG(FIXED_NOISE, FIXED_LABELS).detach().cpu()
                fake_grid = torchvision.utils.make_grid(fake2, padding=2, normalize=True)
                writer.add_image('Generated Images', fake_grid, global_step=ITERS)
                IMG_LIST.append(fake_grid)
        ITERS += 1

    FID = calculate_fretchet(real_cpu, fake, FID_MODEL)
    FID_SCORES.append(FID)
    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFID: %0.4f'
          % (epoch, num_epochs, i, len(dataloader),
             errD.item(), errG.item(), FID))
    writer.add_scalar('FID', FID, global_step=ITERS)

    if epoch % 5 == 0:
        # Saving the full model
        netD_encripted = torch.jit.script(netD)
        netG_encripted = torch.jit.script(netG)

        torch.save(DISC_WEIGHTS + "/netD.pt")
        torch.save(GEN_WEIGHTS + "/netG.pt")

        # Saving the weights every epoch
        torch.save(netD.state_dict(), DISC_WEIGHTS)
        torch.save(netG.state_dict(), GEN_WEIGHTS)

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

from __future__ import print_function

# %matplotlib inline
from torch.utils.tensorboard import SummaryWriter

from FID import *
from utils import *

torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
dataset_name = "fashiongen"
opt = "Adam"
net = "WGAN"


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
elif dataset_name == "fashiongen":
    data_root = "/app/data/fashion_gen/fashion_gen/fashiongen_full_size_train.h5"
    print("Dataset: " + dataset_name)

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

lr = 2e-4

# hyperparam for Adam optimizers
beta1 = 0.5
beta2 = 0.99

# number of gpus available
ngpu = 1

# Critic Iterations
critic_iter = 5

# Weight Clip
weight_clip = 0.01

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
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
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
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training

real_label = 1.
fake_label = 0.

dataset, dataloader = select_dataset(dataset_name=dataset_name, data_root=data_root, transform=transform, dictionary=dictionary,
                                         workers=workers, batch_size=batch_size)

# Setup optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

# Plotting on Tensorboard
print("Creating summary writer...")
directory = "/app/gan_results/"
print("Creating Summary at: " + directory)
writer = SummaryWriter(directory + net + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{lr}")
print("Summary Created on: " + directory + net + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{lr}")

# Showing a batch of training images in tensorboard
real_batch = iter(dataloader)
images, labels = real_batch.next()
img_grid = torchvision.utils.make_grid(images, normalize=True)
writer.add_image('Training images', img_grid)

# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
fid_scores = []
iters = 0

# Preparing Evaluation Metric
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
model = model.cuda()

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
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            label2 = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output1 = netD(real_cpu).view(-1)

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label2.fill_(fake_label)
            # Classify all fake batch with D
            output2 = netD(fake.detach()).view(-1)
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
            fid = calculate_fretchet(real_cpu, fake, model)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFID: %0.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     loss.item(), loss_g.item(), fid))
            ################
            # Discriminator
            ################
            writer.add_scalar('Discriminator Loss', loss, global_step=iters)
            ################
            # Generator
            ################
            writer.add_scalar('Generator Loss', loss_g, global_step=iters)
            ################
            # FID
            ################
            writer.add_scalar('FID', fid, global_step=iters)

        # Save Losses for plotting later
        G_losses.append(loss_g.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                fake_grid = torchvision.utils.make_grid(fake, normalize=True)
                writer.add_image('Generated Images', fake_grid, global_step=iters)

        iters += 1

writer.close()

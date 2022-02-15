from __future__ import print_function

# %matplotlib inline
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
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
net = "CGAN"

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

# Number of workers for dataloader
workers = 2

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
num_epochs = 15

# Learning rate for optimizers
lr = 2e-4

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
beta2 = 0.99

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5 for _ in range(nc)], [0.5 for _ in range(nc)]),
                                ])

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

print("Loading Data...")

dataset, dataloader = select_dataset(dataset_name=dataset_name, data_root=data_root, transform=transform, dictionary=dictionary,
                                     workers=workers, batch_size=batch_size)

print("Data loaded.")

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu, number_classes, embedding_size):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.embedding = nn.Embedding(number_classes, embedding_size)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz+embedding_size, ngf * 8, 4, 1, 0, bias=False),
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
        self.embedding = nn.Embedding(number_classes, image_size*image_size)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc+1, ndf, 4, 2, 1, bias=False),
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
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
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


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
f_labels = cond_labels(number_classes=number_classes)
fixed_noise = torch.randn(96, nz, 1, 1, device=device)
fixed_labels = Variable(torch.LongTensor(f_labels)).view(-1).cuda()


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

# Plotting on Tensorboard
print("Creating summary writer...")
directory = "/app/gan_results/"
print("Creating Summary at: " + directory)
writer = SummaryWriter(directory + net + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{lr}")
print("Summary Created on: " + directory + net + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{lr}")

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

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        y = data[1].to(device)
        labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu, y).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, labels)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
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

        # Output training stats
        if i % 50 == 0:
            fid = calculate_fretchet(real_cpu, fake, model)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFID: %0.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), fid))
            ################
            # Discriminator
            ################
            writer.add_scalar('Discriminator Loss', errD, global_step=iters)
            ################
            # FID
            ################
            writer.add_scalar('FID', fid, global_step=iters)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_labels).detach().cpu()
                writer.add_scalar('Generator Loss', errG, global_step=iters)
                fake_grid = torchvision.utils.make_grid(fake, normalize=True)
                writer.add_image('Generated Images', fake_grid, global_step=iters)

        iters += 1

writer.close()


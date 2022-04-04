from __future__ import print_function

# %matplotlib inline
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, BertModel, AutoTokenizer

from FID import *
from utils import *
from CBN import *

torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
dataset_name = "fashiongen"
opt = "Adam"
net = "T-WGAN"

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
    data_root = "/app/data/celeba/celeba"
    print("Dataset: " + dataset_name)
elif dataset_name == "fashiongen":
    data_root = "/app/data/fashion_gen/fashion_gen/fashiongen_full_size_train.h5"
    print("Dataset: " + dataset_name)

# number of workers for dataloaders
workers = 2
batch_size = [64]
image_size = 64

nc, number_classes, dictionary = data_classes(data_name=dataset_name)

# size of z latent vector
nz = 100
embedding_size = 48
# size of feature maps in discriminator
ndf = 64

# size of feature maps in generator
ngf = 64
bat_s = 64
em_s = 64
num_epochs = 1

learning_rates = [2e-4]

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
    def __init__(self, ngpu, number_classes, embedding_size):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.embedding = torch.nn.Embedding(num_embeddings=number_classes, embedding_dim=embedding_size)
        # input is Z, going into a convolution
        self.convT1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.cbn1 = CBN(batch_size=64, mlp_input=6400, channels=ngf * 8, mlp_hidden=3456, out_size=ngf * 8, height=4, width=4)
        self.act = nn.ReLU(True)
        # state size. (ngf*8) x 4 x 4
        self.convT2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.cbn2 = CBN(batch_size=64, mlp_input=6400, channels=ngf * 4, mlp_hidden=3328, out_size=ngf * 4, height=8, width=8)
        #Relu
        # state size. (ngf*4) x 8 x 8
        self.convT3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.cbn3 = CBN(batch_size=64, mlp_input=6400, channels=ngf * 2, mlp_hidden=3264, out_size=ngf * 2, height=16, width=16)
        # Relu
        # state size. (ngf*2) x 16 x 16
        self.convT4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.cbn4 = CBN(batch_size=64, mlp_input=6400, channels=ngf, mlp_hidden=3232, out_size=ngf, height=32, width=32)
        # Relu
        # state size. (ngf) x 32 x 32
        self.convT5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        self.act2 = nn.Tanh()
        # state size. (nc) x 64 x 64

    def forward(self, input, labels):
        embed = self.embedding(labels).view(-1)
        # Block1
        x = self.convT1(input)
        x, embed = self.cbn1(x, embed)
        x = self.act(x)
        # Block 2 (ngf*8) x 4 x 4
        x = self.convT2(x)
        x, embed = self.cbn2(x, embed)
        x = self.act(x)
        # Block 3 (ngf*4) x 8 x 8
        x = self.convT3(x)
        x, embed = self.cbn3(x, embed)
        x = self.act(x)
        # Block 4 (ngf*2) x 16 x 16
        x = self.convT4(x)
        x, embed = self.cbn4(x, embed)
        x = self.act(x)
        # Block 5 (ngf) x 32 x 32
        x = self.convT5(x)
        x = self.act2(x)
        # state size. (nc) x 64 x 64
        return x


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
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.embedding = torch.nn.Embedding(num_embeddings=48, embedding_dim=1)
        # input is (nc) x 64 x 64
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.cbn2 = CBN(batch_size=64, mlp_input=64, channels=ndf * 2, mlp_hidden=96, out_size=ndf * 2, height=16,
                        width=16)
        # features and lstm_embed
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1,
                               bias=False)
        self.cbn3 = CBN(batch_size=64, mlp_input=64, channels=ndf * 4, mlp_hidden=160, out_size=ndf * 4, height=8, width=8)
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1,
                               bias=False)
        self.cbn4 = CBN(batch_size=64, mlp_input=64, channels=ndf * 8, mlp_hidden=288, out_size=ndf * 8, height=4, width=4)
        # state size. (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        self.activation = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, input, labels):
        embed = self.embedding(labels).view(-1)
        # Block 1 (nc) x 64 x 64
        x = self.conv1(input)
        x = self.activation(x)
        # Block 2 (ndf) x 32 x 32
        x = self.conv2(x)
        x, embed = self.cbn2(x, embed)
        x = self.activation(x)
        # Block 3 (ndf*2) x 16 x 16
        x = self.conv3(x)
        x, embed = self.cbn3(x, embed)
        x = self.activation(x)
        # Block 4 (ndf*4) x 8 x 8
        x = self.conv4(x)
        x, embed = self.cbn4(x, embed)
        x = self.activation(x)
        # Block 5 (ndf*8) x 4 x 4
        x = self.conv5(x)
        return x


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
f_labels = cond_labels(number_classes=number_classes)
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
fixed_labels = Variable(torch.LongTensor(f_labels)).view(-1).cuda()

# Establish convention for real and fake labels during training

real_label = 1.
fake_label = 0.

# Preparing Evaluation Metric
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model_eval = InceptionV3([block_idx])
model_eval = model_eval.cuda()

'''# Loading the Transformer
config = BertConfig()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_model = BertModel(config)'''
for bs in batch_size:
    dataset, dataloader = select_dataset(net=net, dataset_name=dataset_name, data_root=data_root, transform=transform,
                                         dictionary=dictionary,
                                         workers=workers, batch_size=bs)

    for lr in learning_rates:
        # Training Loop
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        fid_scores = []
        iters = 0

        # Setup RMSprop optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

        print("Creating summary writer...")
        directory = "/app/gan_results/"
        print(directory)
        print("Creating Summary at: " + directory)
        writer = SummaryWriter(
            directory + net + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{bs}" + "/" + f"{lr}")
        print(
            "Summary Created on: " + directory + net + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{bs}" + "/" + f"{lr}")

        # Showing a batch of training images in tensorboard
        real_batch = iter(dataloader)
        images, labels = real_batch.next()
        img_grid = torchvision.utils.make_grid(images, normalize=True)
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
                    labels_batch = list(data[1])
                    labels_batch = change_label(labels_batch, dictionary).to(device)
                    b_size = real_cpu.size(0)

                    # Forward pass real batch through D
                    output1 = netD(real_cpu, labels_batch).view(-1)

                    # Train with all-fake batch
                    # Generate batch of latent vectors
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    # Generate fake image batch with G
                    fake = netG(noise, labels_batch)
                    # Classify all fake batch with D
                    output2 = netD(fake.detach(), labels_batch).view(-1)
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
                output = netD(fake, labels_batch).view(-1)
                # Calculate G's loss based on this output
                loss_g = -torch.mean(output)
                # Calculate gradients for G
                loss_g.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    fid = calculate_fretchet(real_cpu, fake, model_eval)
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
                if (iters % 500 == 0):
                    with torch.no_grad():
                        fake = netG(fixed_noise, fixed_labels).detach().cpu()
                        fake_grid = torchvision.utils.make_grid(fake, normalize=True)
                        writer.add_image('Generated Images', fake_grid, global_step=iters)
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
        writer.add_hparams({"lr": lr, "bsize": bs}, {"FID": fid})
        writer.close()

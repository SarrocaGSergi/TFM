from __future__ import print_function

import torch
import torchvision
import argparse
import random
from torch import optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer

from DCGAN_arch import *
from utils import *

torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 899
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(f"DEVICE: {device}")

# Defining Dataset, Optimizer & Architecture
dataset_name = "fashiongen"
opt = "Adam"
network = "ST-DCGAN"


# Defining some variables
DATA_ROOT = get_data_root(database_name=dataset_name)

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
embedding_size = 100
# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 20
embedding_size = 768
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

dataset, dataloader = load_dataset(net=network, dataset_name=dataset_name, data_root=DATA_ROOT, transform=TRANSFORM,
                                   dictionary=dictionary, workers=n_workers, batch_size=batch_size)

print("Data loaded.")

# Load Architecture

# Create the Discriminator
netD = Discriminator(ngpu, nc, ndf, embedding_size).to(device)

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
netG = Generator(ngpu, nz, nc, ngf, embedding_size).to(device)

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
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator

FIXED_LABELS = ["BACKPACKS", "BAG ACCESSORIES", "BELTS & SUSPENDERS", "BLANKETS", "BOAT SHOES & MOCCASINS",
                "BOOTS", "BRIEFCASES", "CLUTCHES & POUCHES", "DRESSES", "DUFFLE & TOP HANDLE BAGS",
                "DUFFLE BAGS", "ESPADRILLES", "EYEWEAR", "FINE JEWELRY", "FLATS", "GLOVES",
                "HATS", "HEELS", "JACKETS & COATS", "JEANS", "JEWELRY", "JUMPSUITS",
                "KEYCHAINS", "LACE UPS", "LINGERIE", "LOAFERS", "MESSENGER BAGS",
                "MESSENGER BAGS & SATCHELS", "MONKSTRAPS", "PANTS", "POCKET SQUARES & TIE BARS",
                "POUCHES & DOCUMENT HOLDERS", "SANDALS", "SCARVES", "SHIRTS", "SHORTS",
                "SHOULDER BAGS", "SKIRTS", "SNEAKERS", "SOCKS", "SUITS & BLAZERS", "SWEATERS",
                "SWIMWEAR", "TIES", "TOPS", "TOTE BAGS", "TRAVEL BAGS", "UNDERWEAR & LOUNGEWEAR", "BOOTS", "BOOTS",
                "BOOTS", "BOOTS", "BOOTS", "BOOTS", "BOOTS", "BOOTS", "BOOTS", "BOOTS", "BOOTS", "BOOTS", "BOOTS",
                "BOOTS", "BOOTS", "BOOTS"]


FIXED_LABELS2 = ["Fitted blazer with notch lapels and single button closure. Front welt pockets, decorative buttons on cuffs and back slit. Lined. The female model in the photo is 5'10 and wears a size 38.",
"Oversize double breasted blazer. Model with fine peak lapels, one top pocket and front pockets with flaps. Lined.", "Oversize blazer with notch lapels, button closure and decorative top pocket. Straight single-breasted model with front pockets trimmed with flaps. Lined. Female model pictured is 5'9 and is wearing a size XL.",
"Double-breasted blazer with notch lapels, one top bias pocket and front flap welt pockets. Straight cut model with decorative buttons on cuffs and back opening. Lined.",
"Relaxed cut blazer with three-quarter sleeves. Straight cut with notch lapels and front pockets with flaps. Without zipper. Lined.",
"Relaxed cut blazer with three-quarter sleeves. Straight cut with notch lapels and front pockets with flaps. Without zipper. Lined.",
"Fitted blazer made of fabric with single button closure. Model with inset front pockets, covered button on cuffs and opening at the back. Lined.",
"Single breasted blazer in stretch fabric with notch lapels with decorative buttonhole and two front buttons, one top pocket, front pockets with flaps and one inside pocket. Decorative buttons on cuffs and back slit. Lined. Slim fit - slim fit through the chest and waist, with slightly narrower sleeves for a slim silhouette. The male model in the photo is 6'2 and wears a size 44.",
"Single breasted blazer in linen with notch lapels with decorative buttonhole. Model with two front buttons, one top pocket, front pockets with flap, two inside pockets, decorative buttons on cuffs and back opening. Lined. Slim fit - fitted at chest and waist, with slightly narrower sleeves for a refined silhouette. Slim fit.",
"Double-breasted linen blazer with peak lapels and decorative buttonhole. Standard cut model with top pocket, front pockets with flap and two inside pockets. Decorative buttons on cuffs and double vent at back.",
"Synthetic leather boots with ankle shaft, laces in front and belt loop at the back. Synthetic leather insole and thick sole. Heel 3 cm.",
"Studio Collection. Vegea vegetable leather clogs with metal studs. Model with thick sole and partially covered square heel. Lining and padded insole in vegetable Vegea leather.",
"Slightly padded slippers in quilted synthetic leather. Wide strap with hook-and-loop fastener and padded synthetic leather insole. Ribbed sole.",
"Synthetic leather moccasins with decorative bow in front. Satin lining and synthetic leather insole. Thick sole with design. Heel 6 cm.",
"Synthetic suede high top sneakers with mesh, knit and synthetic leather sections. Padded collar, laces with hooks on the upper and loop at the back. Lining and insole in piqué. Thick sole with design. Sole thickness 5 cm.",
"Synthetic leather moccasins with stitching and decorative bow in front. Satin lining and synthetic leather insole. Thick sole with design. Heel 4 cm.",
"Synthetic leather boots with ankle shaft, laces in front and belt loop at the back. Synthetic leather insole and thick sole. Heel 3 cm.",
"Synthetic leather Chelsea boots with half shaft, elastic sides and front and back straps. Satin lining and synthetic leather insole. Thick sole with ribbed design.",
"Synthetic leather sandals with frontal straps and bracelet with hook-and-loop fastener. Synthetic leather insole and thick sole with ribbed design. Thickness of the sole 6 cm.",
"Synthetic leather mules with square toe, asymmetrical instep strap and covered square heel. Synthetic leather lining and insole. Heel 10 cm.",
"Recycled polyester bikini bottoms with ruffle trim. Low-waisted model with semi-covered back. Fully lined.",
"Ribbed swimsuit with adjustable straps and removable padded cups that shape and provide good support. Semi-covered back. Lined.",
"Extra tight-fitting swimsuit with shaping effect at the waist. Model with V-neckline, adjustable straps and removable padded cups that shape the bust and provide good support. Decorative knot in front and semi-covered back. Fully lined.",
"Triangle bikini top in recycled polyester with overcast ruffle trim. Adjustable thin straps with multi-function back closure and removable padded cups that shape the bust and provide good support. Metal back closure. Lined. No underwire.",
"Studio Collection. Triangle bikini top with leopard print. Model with spaghetti straps and thin strap at the bottom, both elastic and adjustable. Gold metal clasp at the back.",
"Printed knee-length swim trunks with coated elastic and drawstring waist, side pockets and insert pocket with hook-and-loop closure at the back. Soft mesh inner briefs.",
"Fabric swimsuit with elastic waistband and drawstring, bias pockets and a back pocket with hook-and-loop closure. Mesh inner briefs.",
"Fabric swimsuit with elastic waistband and drawstring, bias pockets and a back pocket with hook-and-loop closure. Mesh inner briefs.",
"Printed knee-length swim trunks with coated elastic and drawstring waist, side pockets and insert pocket with hook-and-loop closure at the back. Soft mesh inner briefs.",
"Recycled polyamide swimsuit with elastic waistband and drawstring, bias pockets and a back pocket with hook-and-loop closure. Mesh inner briefs.",
"Five-pocket jeans in stretch cotton denim. Standard waist model with fitted legs and zipper and button closure.",
"Five-pocket jeans in stretch cotton denim. Standard waist model with extra-fitted legs and zipper closure with button.",
"Five-pocket jeans in thick cotton denim. Standard waist, zipper closure with button and wide leg for a loose fit.",
"Five-pocket jeans in stretch cotton denim. Standard waist model with fitted legs and zipper and button closure.",
"Five-pocket denim jeans with standard waistband, zipper and button closure and slim leg. Made with Lycra Freefit technology that provides optimum stretch, maximum mobility and excellent comfort.",
"Five-pocket jeans in washed cotton denim. High-waisted model with wide straight legs and zipper closure with button.",
"Five-pocket jeans in stretch cotton denim. Low-waisted model with zipper closure with button and wide legs with flared hem.",
"Five-pocket jeans in stretch cotton denim. Low-waisted model with zipper and button closure, back pockets with flap and button, and flared hems.",
"Five-pocket jeans in washed cotton denim. High-waisted model with wide straight legs and zipper closure with button.",
"Five-pocket ankle jeans in washed stretch denim. High-waisted model with zipper closure with button and fitted legs.",
"Cotton twill bucket hat with small motif on the front and embroidered eyelets on the sides. Anti-sweat band and cotton lining.",
"Cotton twill cap with adjustable metal clasp at the back. Cotton sweatband.",
"Bucket hat with text motif. Elastic drawstring for optimal fit. Cotton sweatband and mesh lining.",
"Cotton twill cap with adjustable metal clasp at the back. Cotton sweatband.",
"Cotton twill and mesh cap with embroidered appliqué on front and adjustable plastic closure on back. Cotton sweatband.",
"Regenerated leather belt with metal buckle. Width 3.5 cm.",
"Regenerated dark blue leather belt with metal buckle. Width 3.5 cm.",
"Brown leather belt with metal buckle. Width 3.3 cm.",
"Braided elastic belt with synthetic leather details and metal buckle. Width approx. 3.5 cm.",
"Black leather belt with metal buckle. Width 3.3 cm.",
"Mini shoulder bag with small fabric appliqué on the front, top zipper with double slider and removable adjustable strap with buckles and plastic snap hook. One exterior zippered compartment and one interior compartment. Lined. Width 4 cm. Length 15.5 cm. Height 20.5 cm.",
"Small weekend bag with two handles, top zipper and removable adjustable shoulder strap. One exterior and one interior zippered compartment. Metal buckles on the sides to adjust the size. Lined. Width 22 cm. Length 48 cm. Height 31 cm.",
"Fabric backpack with adjustable padded handle and straps. Top zipper, one exterior zippered compartment and two interior compartments, one for laptop and one smaller zippered compartment. Padded back. Lined. Width 13 cm. Length 30 cm. Height 43 cm.",
"This shopper bag is made of partially recycled cotton canvas. Model with double handle in contrasting color that continues downwards, an inner compartment with zipper and base in contrasting color. Width 35 cm. Height 47 cm. Length 55 cm. Pocket measures 20x21 cm.","Multifunctional canvas bag that can be used as a backpack and as a weekend bag. Two top handles and zipper closure. Lightly padded adjustable straps with snap closure and side mesh concealment pocket. Exterior zippered compartment and interior zippered pocket on flap. Width approx. 30 cm. Height approx. 30 cm. Length approx. 52 cm.",
"Fabric shoulder bag with zippered top and removable adjustable shoulder strap with buckle and metal carabiners. One exterior compartment with hidden zipper and one interior mesh compartment. Lined. Width 7 cm. Height 16 cm. Length 23 cm.",
"Fabric fanny pack with large zippered front compartment, zippered back compartment and open inner compartment. Adjustable strap with plastic snap closure. Lined. Width approx. 6 cm. Height approx. 15 cm. Length approx. 18 cm.",
"Bum bag with one large zippered compartment in front, one zippered compartment in back and one inner compartment. Adjustable strap with plastic snap closure. Lined. Measures 5x18x32 cm.",
"Weekend bag in synthetic leather with two handles, removable shoulder strap, zipper on top and two compartments inside. Support studs on the base. Lined. Measures approx. 22x31x48 cm.",
"Multifunctional ripstop bag that can be used as a backpack or tote bag. Double handle model with zippered closure, lightly padded adjustable straps with snap closure and back compartment for strap storage. Zippered front compartment, open mesh side compartments and interior zippered pocket for easy folding and storage. Practical strap to carry it folded. Lined. Width approx. 8 cm. Length approx. 34 cm. Height approx. 35 cm.",
"Metal earrings and ear cuffs. Six hoop earrings. Seven ear cuffs. Made with recycled zinc.",
"Gold plated metal chain anklets in different designs. Adjustable length and spring ring closure.",
"Round sunglasses with plastic frame, lenses and temples. Polarized tinted lenses with UV protection for color clarity and contrast while eliminating glare. Supplied in a fabric case with drawstring.",
"Sunglasses with plastic frame. Plastic tinted lenses with UV protection."]
print(len(FIXED_LABELS2))

# FIXED_LABELS = Variable(torch.tensor(labs)).cuda()
# FIXED_LABELS = training_one_hot(FIXED_LABELS, dictionary).to(device)
# FIXED_LABELS = change_labels(dictionary, FIXED_LABELS).to(device)

FIXED_NOISE = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, beta2))


# Plotting on Tensorboard
print("Creating summary writer...")
directory = "/app/gan_results/new_cbn/Sentence-Transformer/"
OUT_PATH = directory + network + "/" + dataset_name + "/" + f"{num_epochs}" + "/" + opt + "/" + f"{batch_size}" + "/" + f"{learning_rate}"
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
# FIXED_IMAGES, TARGETS = projector_img_labs(dataset=dataset, b_size=100, n_workers=2)

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
        y = list(data[1])
        encoded_tokens = tokenizer(y, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_tokens)
        sequence_embeddings = mean_pooling(model_output, encoded_tokens['attention_mask']).to(device)

        labels = torch.full((bat_size,), real_label, dtype=torch.float, device=device)

        output = netD(real_cpu, sequence_embeddings).view(-1)

        # Calculate loss on all-real batch
        errD_real = criterion(output, labels)
        # Calculate gradients for D in backward pass
        errD_real.backward(retain_graph=True)
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(bat_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise, sequence_embeddings)
        labels.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach(), sequence_embeddings).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, labels)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward(retain_graph=True)
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
        output = netD(fake, sequence_embeddings).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, labels)
        # Calculate gradients for G
        errG.backward(retain_graph=True)
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Output training stats
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item()))
            # Discriminator
            writer.add_scalar('Discriminator Loss', errD, global_step=ITERS)
            # Generator
            writer.add_scalar('Generator Loss', errG, global_step=ITERS)
            # Projector
            '''
            features = FIXED_IMAGES.reshape(FIXED_IMAGES.shape[0], -1)
            class_labels = [dictionary[int(lab)] for lab in TARGETS]
            writer.add_embedding(features, metadata=class_labels, label_img=FIXED_IMAGES, global_step=ITERS)
            '''

        # Check how the generator is doing by saving G's output on fixed_noise
        if (ITERS % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                FIXED_ENCODED = tokenizer(FIXED_LABELS2, padding=True, truncation=True, return_tensors='pt').to(device)
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
          % (epoch, num_epochs, i, len(dataloader),
             errD.item(), errG.item(), FID))
    writer.add_scalar('FID', FID, global_step=ITERS)


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

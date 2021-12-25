import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from PIL import Image
from models import *

train_root = "/app/data/fashion_gen/fashion_gen/fashiongen_full_size_train.h5"

dictionary = {0: "BACKPACKS",
                      1: "BAG ACCESSORIES",
                      2: "BELTS & SUSPENDERS",
                      3: "BLANKETS",
                      4: "BOAT SHOES & MOCCASINS",
                      5: "BOOTS",
                      6: "BRIEFCASES",
                      7: "CLUTCHES & POUCHES",
                      8: "DRESSES",
                      9: "DUFFLE & TOP HANDLE BAGS",
                      10: "DUFFLE BAGS",
                      11: "ESPADRILLES",
                      12: "EYEWEAR",
                      13: "FINE JEWELRY",
                      14: "FLATS",
                      15: "GLOVES",
                      16: "HATS",
                      17: "HEELS",
                      18: "JACKETS & COATS",
                      19: "JEANS",
                      20: "JEWELRY",
                      21: "JUMPSUITS",
                      22: "KEYCHAINS",
                      23: "LACE UPS",
                      24: "LINGERIE",
                      25: "LOAFERS",
                      26: "MESSENGER BAGS",
                      27: "MESSENGER BAGS & SATCHELS",
                      28: "MONKSTRAPS",
                      29: "PANTS",
                      30: "POCKET SQUARES & TIE BARS",
                      31: "POUCHES & DOCUMENT HOLDERS",
                      32: "SANDALS",
                      33: "SCARVES",
                      34: "SHIRTS",
                      35: "SHORTS",
                      36: "SHOULDER BAGS",
                      37: "SKIRTS",
                      38: "SNEAKERS",
                      39: "SOCKS",
                      40: "SUITS & BLAZERS",
                      41: "SWEATERS",
                      42: "SWIMWEAR",
                      43: "TIES",
                      44: "TOPS",
                      45: "TOTE BAGS",
                      46: "TRAVEL BAGS",
                      47: "UNDERWEAR & LOUNGEWEAR"}

ngpu = 1
torch.cuda.empty_cache()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

'''save_dir = '/app/vscode/'
print('Creating SummaryWriter')
writer = SummaryWriter(save_dir + 'test6')
print('SummaryWriter Created')'''
image_size = 256
nc = 3

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5 for _ in range(nc)], [0.5 for _ in range(nc)]),
                                ])

dataset = FashionGen(train_root=train_root, transform=transform, dictionary=dictionary)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)


print('Showing Images at tensorboard')
dataiter = iter(dataloader)
data, labels = dataiter.next()
print(labels)
#tokens = tokenizer(l_d, padding=True, truncation=True, return_tensors='pt')
#grid = torchvision.utils.make_grid(data, normalize=True)
# writer.add_image('Training images', grid)
#plt.imsave(f"{save_dir}"+"grid.jpg", np.transpose(grid.numpy(), (1, 2, 0)))

print('Finished')



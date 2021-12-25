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

data_root = "../app/data/vscode"

directory = "../app/vscode/new_folder"

if not os.path.exists(directory):
    print("Creating new directory")
    os.mkdir(directory)
    print("Created")
elif os.path.exists(directory):
    print("Already exists")

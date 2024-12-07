import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

# for config
import yaml
import sys
import os
from tqdm import tqdm
# adding paths
sys.path.append(os.path.join('DiT'))
sys.path.append(os.path.join('DDPM'))

# for model
import transformer # main-model
import sampler # DDPM

with open('./config/config.yaml') as file:
    config = yaml.safe_load(file)


'''Training Model (DiT)
Now, we will be generating batches of images that we wish to
predict the noise from.

We will need to have a pre-trained VAE to work in the latent space,
and have a DDPM noiser that can noise at any arbitrary timestep
and generate the ground-truth noise that we compare against.

Our loss function is:
(See (DDPM))

We do not predict the variance like the original paper,
we have it fixed.
'''

# setting up model
model = transformer.DiT(config['dit_params'])


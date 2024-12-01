import torch
import torch.nn as nn
from torch.nn import function as F


'''VAE
Here, we are training our Variational
autoencoder, which consists of an encoder and 
a decoder.

We'll use our encoder to encode the ground-truth
latents, and our decoder if we want to sample from
our model.

However, our VAE will be decoupled from the rest
of the training itself.

'''

# our encoder, makes latent rep.
# should be able to deeply understand
# images


class VAEncoder(nn.Module): 
    # takes in:
    def __init__(self, im_size):
        super().__init__()
# downloading from diffusers.
from diffusers.models import AutoencoderKL
import torch.nn as nn
import torch


# create custom-vae class
'''VAE
When we create the VAE, we utilize hugging-faces
pre-built VAE, and split the decoding, and encoding part.

We acess the latent_dist params with .mean and .logvar
and sample from decoder with .sample.
'''
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained('stabilityai/sdxl-vae')
    
    # take encoding and reparam
    def encode_reparam(self,x):
        # B,8,32,32 -> 2 of B,4,32,32
        # take in encoder...
        input = self.vae.encode(x)
        mean,logvar = input["latent_dist"].mean,input["latent_dist"].logvar
        noise = torch.randn_like(logvar)
        latent = mean + noise*(logvar*0.5).exp()
        return latent
    
    def decode(self,x):
        return self.vae.decode(x).sample
    
    def forward(self,x):
        x = self.encode_reparam(x)
        x = self.decode(x)
        return x
    
class VAE_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained('stabilityai/sdxl-vae')
        
    def forward(self,x):
        # B,8,32,32 -> 2 of B,4,32,32
        # take in encoder...
        input = self.vae.encode(x)
        mean,logvar = input["latent_dist"].mean,input["latent_dist"].logvar
        noise = torch.randn_like(logvar)
        latent = mean + noise*(logvar*0.5).exp()
        return latent
    
# returning
def get_vae():
    full_vae = VAE()
    encoding_vae = VAE_encoder()
    return full_vae, encoding_vae


# test: works!
t = torch.randn([1,3,32,32])
vae,encoder = get_vae()
t = encoder(t)
print(vae.decode(t).shape)
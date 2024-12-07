import torch
from torch import nn as nn


'''DiT sampler
We are relying on the mathematical formulation of diffusion models
to train our model, and preform inference.

DDPM (training)
we will be sampling from random timesteps,
and predicting the ground-truth noise with 
MSE loss

'''
class DDPM:
    def __init__(self,steps,betas):
        self.betas = torch.linspace(betas[0],betas[1],steps)
        self.alphas = 1-self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=0)
        self.steps= steps
        
    def add_random_noise(self,batch):
        # for each batch, add-noise,
        # and return noisy image + noise
        input_shape = batch.shape
        # easy to deal with
        batch = batch.view(input_shape[0],-1)
        noise = torch.randn_like(batch)
        # broadcast across dim=1
        steps = torch.randint(0,self.steps,size=(batch.shape[0],))
        
        # selecting from alpha bars
        # B, 1
        alpha_bars = self.alphas_cumprod[steps].view(-1,1)
        
        # adding according to noising process
        # std dev and mean
        noised_images = alpha_bars**0.5 * batch + (1-alpha_bars)**0.5 * noise
        noised_images = noised_images.view(input_shape)
        
        # step doesn't matter; predicting
        # same ground-truth noise!
        noise = noise.view(input_shape)
        return noised_images, noise
    
    
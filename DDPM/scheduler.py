import torch
from torch import nn as nn



'''Scheduler
Building out DDPM to attempt to learn to sample
from the distribution of our dataset gradually.

We will define our noising process parameters:
alphas_cumprod, alphas, betas, betas_cumprod

Furthermore, we will define:
- Noising process to give noisy image for training
given a certain timestep
- Ground-truth noise for an image given the original, 
and the timestep + noisy (for loss-function)
'''


class Scheduler:
    # Normal Scheduler used in DDPM
    # takes in: num_timesteps, beta start/end
    def __init__(self,num_timesteps, beta_start,beta_end):
        
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # all betas, linear schedule
        self.betas = torch.linspace(beta_start,beta_end,num_timesteps)
                
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=-1)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        
        self.sqrt_betas_cumprod = torch.sqrt(1-self.alphas_cumprod)
    
    # add noise to image at given timestep
    def add_noise(self, original, noise, t):
        # send to device
        # Size: B
        alpha_bar = self.sqrt_alphas_cumprod.to(original.device)[t].reshape(original.shape[0])
        beta_bar = self.sqrt_betas_cumprod.to(original.device)[t].reshape(original.shape[0])
        
        while len(alpha_bar.shape) < original.shape:
            # adding dimensions until (B,1,1,...)
            alpha_bar.unsqueeze(-1)
            beta_bar.unsqueeze(-1)
        
        # re-param trick, where x_t is 
        # alpha_bar * x_0 + beta_bar * eps_n
        # (B,C,H,W) * (B,1,1,1) + (B,1,1,1)
        noised = original*alpha_bar + beta_bar*noise
        
        return noised
    
    # getting our prev. timestep given
    # the noise
    def sample_prev_timestep(self,xt,pred,t):
        # we are sampling using the formula
        # for the mean in DDPM
        betas = self.betas.to(device=xt.device)
        alphas = self.alphas.to(device=xt.device)
        alphas_cum_prod = self.alphas_cumprod.to(device=xt.device)
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device=xt.device)
        sqrt_betas_cumprod = self.sqrt_betas_cumprod.to(device=xt.device)
        
        
        x_zero = (xt - sqrt_betas_cumprod[t]*pred) / sqrt_alphas_cumprod[t]
        
        mean = xt / sqrt_alphas_cumprod[t]
        mean -= pred*(betas[t]*sqrt_betas_cumprod[t]) / ((1-alphas_cum_prod[t])*alphas[t]**0.5)
        
        # if our timestep is the final, we should
        # return our mean, without new noisee
        # also returning the prediction for other use
        if t == 0:
            return mean, x_zero
        else:
            var = (betas[t])*(1-alphas_cum_prod[t-1]) / (sqrt_betas_cumprod[t])
            std = var**0.5
            noise = torch.rand_like(pred,device=pred.device)
            return mean + std*noise, x_zero
        
    
        
        

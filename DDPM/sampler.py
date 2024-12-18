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
# DDPM w/ hybrid-loss 

class DDPM:
    def __init__(self,steps,betas,scale,loss2_weight=0.001,device='cpu'):
        
        self.device=device
        
        self.loss2_weight = loss2_weight
        self.betas = torch.linspace(betas[0]**(1/scale),betas[1]**(1/scale),steps,device=device) ** scale
        self.alphas = 1-self.betas
        self.alphas_sqrt = torch.sqrt(self.alphas)
        
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=0)
        self.betas_cumprod = 1 - self.alphas_cumprod
        
        # for quick-compute
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_betas_cumprod = torch.sqrt(self.betas_cumprod)
        
        
        self.steps= steps
        
    def add_random_noise(self,batch):
        # for each batch, add-noise,
        # and return noisy image + noise
        input_shape = batch.shape
        # easy to deal with
        batch = batch.view(input_shape[0],-1)
        noise = torch.randn_like(batch,device=self.device)
        # broadcast across dim=1
        steps = torch.randint(0,self.steps,size=(batch.shape[0],),device=self.device)
        
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
        return noised_images,steps, noise
    
    def hybrid_loss(self,model_out,noise,t,noised_image):
        # getting both of our outputs
        # B, 2, c, h,w -> 2x, b,c,h,w
        # will need to use .view for shape-matching,
        # to broadcast for each b-dim
        pred_noise,pred_var = model_out.chunk(2,dim=1)
        pred_noise,pred_var = pred_noise.squeeze(1),pred_var.squeeze(1)
        
        # simple-loss
        l_simple = torch.mean((pred_noise-noise)**2)
        
        # calculating ground-truth mean and
        # predicted-mean (same but diff.-noise)
                
        pred_mean = (1/ self.alphas_sqrt[t].view(-1,1,1,1)) * \
            (noised_image-(self.betas[t].view(-1,1,1,1)/self.sqrt_betas_cumprod[t].view(-1,1,1,1))*pred_noise)
        real_mean = (1/ self.alphas_sqrt[t].view(-1,1,1,1)) * \
            (noised_image-(self.betas[t].view(-1,1,1,1)/self.sqrt_betas_cumprod[t].view(-1,1,1,1))*noise)
        
        # lower/upper bds of beta
        upper_beta = self.betas[t] # also 'ground-truth'
        lower_beta = self.betas[t] * self.betas_cumprod[t-1] / self.betas_cumprod[t]
        pred_var = torch.exp(torch.log(upper_beta).view(-1,1,1,1)*pred_var + torch.log(lower_beta).view(-1,1,1,1)*(1-pred_var))
        
        # full-loss,
        # should broadcast to B,shape_img,
        # stopgrad on mean, and average all of 
        # loss-parts together?
        dkl_loss = torch.mean((0.5 * (lower_beta.view(-1,1,1,1)/pred_var + (((pred_mean-real_mean)**2).detach())/pred_var - 1 + torch.log(pred_var/lower_beta.view(-1,1,1,1)))))
        hybrid_loss = dkl_loss*self.loss2_weight + l_simple
        
        with torch.no_grad():
            self.loss_ratio = (dkl_loss*self.loss2_weight / l_simple)
        
        return hybrid_loss
    
    def hybrid_loss_ratio(self):
        return self.loss_ratio.item()
    
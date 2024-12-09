import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import torchvision
# for config
import yaml
import sys
import os
from tqdm import tqdm
# adding paths
sys.path.append(os.path.join('DiT'))
sys.path.append(os.path.join('DDPM'))
sys.path.append(os.path.join('data'))

# for model
import transformer # main-model
import sampler # DDPM
import dataloader # for loading data

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

# hyper params:
num_iters = 500
batch_size=16
device = 'cuda'
load = False
    

##### Cust dataset
latent_path = os.path.join('data','latent_folder_sdxl')
label_path = os.path.join('data','label_folder')
custom_data = dataloader.ImageDataset(transform=None,im_path='raw_images',label_folder=label_path,im_extension='jpeg',use_latents=True,latent_folder=latent_path)
########

# setting up model
model = transformer.DiT(config['dit_params'])
# loading if specified
if load:
    model.load_state_dict(torch.load('weights.pt',model))
model.to(device)
# betas = begin noise/end, just follow authors
# this is how the paper defines it...
ddpm_sampler = sampler.DDPM(steps=1000,betas=[0.0001,0.02],device=device)



############ Training #########
def get_batch(data,batch_size=batch_size):
    indices = torch.randint(0,len(data),(batch_size,))
    
    # x and y ind of batch
    x = torch.stack([data[i][0][0] for i in indices],dim=0)
    y = torch.stack([torch.tensor(data[i][1]) for i in indices],dim=0)
    return x.to(device),y.to(device)

# optimizer for quick convergence
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.999),weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,100,0.2)
# loss-function = MSE for noise-vector
criterion = nn.MSELoss()


losses = []
# for now, no class-labels
for i in range(num_iters):
    # getting batch
    batch,y = get_batch(custom_data)
    
    # (b,c,h,w), (b), (b,c,h,w)
    noised_images,timesteps,noise = ddpm_sampler.add_random_noise(batch)
    
    # model expects batch and time-step
    noise_pred = model(noised_images,timesteps)
    
    # backward-pass
    loss = criterion(noise_pred,noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())
    if i % 2 ==0:
        print(f"Iter {i}, loss: {loss}")
    if i % 200 ==0:
        print(f"weights saved")
        torch.save(model,'weights.pt')
        
        
### plotting loss
import matplotlib.pyplot as plt
plt.plot(losses)
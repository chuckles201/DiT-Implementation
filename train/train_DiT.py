import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import torchvision
# for config
import yaml
import sys
import os
from tqdm import tqdm
import numpy as np
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

###############################################################

'''Multi-GPU support

adding multi-gpu support, so 
each gpu does its own random-batch,
and losses are added-together
before we do a step!'''

# required modules
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# intializes ddp
def ddp_setup(rank,world_size):
    # rank: identifies
    # world-size: all
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = 12355 # random-port
    init_process_group(backend="nccl",rank=rank,world_size=world_size) # nvidia

# 1. Send model/all other things to current-gpu
# 2. wrap moddel w/DDP (distributed data-parallel)

# 3. Each model does a different part of data
# 4. We average gradients between model before steps!
###############################################################


# hyper params:
num_iters = 1000000
batch_size=16
device = 'cuda'    
betas = [1e-4,2e-2]

##### Cust dataset
latent_path = os.path.join('data','latent_folder_sdxl')
label_path = os.path.join('data','label_folder')
custom_data = dataloader.ImageDataset(transform=None,im_path='raw_images',label_folder=label_path,im_extension='jpeg',use_latents=True,latent_folder=latent_path)
########


# LOADING MODEL #
model = transformer.DiT(config['dit_params'])

load_model = input("Load model? (y/n)\n")
if load_model == "y".lower():
    model.load_state_dict(torch.load('./weights.pt'))


# betas = begin noise/end, follwing 
# authors
ddpm_sampler = sampler.DDPM(steps=2000,
                            betas=betas,
                            scale=1,
                            loss2_weight=1.5e-3,
                            device='cuda')


############ Training #########
def get_batch(data,batch_size=batch_size):
    indices = torch.randint(0,len(data),(batch_size,))
    
    # x and y ind of batch
    x = torch.stack([data[i][0][0] for i in indices],dim=0)
    y = torch.stack([torch.tensor(data[i][1]) for i in indices],dim=0)
    return x.to(device),y.to(device)

# optimizer for quick convergence
optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(0.9,0.999),weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,100,0.998)
# loss-function = MSE for noise-vector

model.train().to(device)
losses = []
# for now, no class-labels
import time
start = time.time()
for i in range(num_iters):
    # getting batch
    batch,y = get_batch(custom_data)
    
    # (b,c,h,w), (b), (b,c,h,w)
    noised_images,timesteps,noise = ddpm_sampler.add_random_noise(batch)
    
    # model expects batch and cond (t and label)
    model_out = model(noised_images,timesteps,y)
    
    # backward-pass
    # using our hybrid-loss from paper:
    # 'improved DDPM's
    loss = ddpm_sampler.hybrid_loss(model_out,noise,timesteps,noised_images)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    losses.append(loss.item())
    
    checkpoint = 10
    torch.cuda.synchronize()
    if i % checkpoint ==0:
        print(f"Iter {i} | Time:{time.time()-start} | Avg. loss: {np.mean(losses[-checkpoint:i])} ")
        print(f"Hybrid-ratio: {ddpm_sampler.hybrid_loss_ratio()}")
        start = time.time()
        
    if i % 100 ==0:
        print(f"weights saved")
        torch.save(model.state_dict(),'weights.pt')
        grads=[]
        for name,param in model.named_parameters():
            print(f"Name: {name} ||| Grad mean: {torch.mean(param.grad)}, std:{torch.std(param.grad)}\n")
            grads.append(torch.std(param.grad).item())
        
### plotting loss
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
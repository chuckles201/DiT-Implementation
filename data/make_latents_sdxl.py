import torch
import PIL
from dataloader import ImageDataset
import os
from tqdm import tqdm

''' Storing Latents
This is modified for storing
the encoder outputs of SD's
VAE, and does not take batch size
as an argument, and saves encoder
outputs (dict).
'''

def store_latents(dataset,path,model):
    # takes entire dataset and maps to 
    # a specific path for latent images.
    # returns the path of the representations
    model.eval()# turn off training!
    latent_outputs = []
    full_path = os.path.join(path,'latent_storage.pt')
    
    # creating folder
    if not os.path.exists(path):
        os.makedirs(path)
    
    # if already there
    if os.path.exists(full_path):
        print("Mapping already exists, skipping latent-storage")
       
    # creating file with latents 
    else: # NEED for cuda memory
        with torch.no_grad(): 
            # take all images.
            ra = tqdm(range(len(dataset)))
            for i in ra:
                # appending latent-outputs
                # passing-thru labels.
                # add extra-b dim
                sample = dataset[i][0].unsqueeze(0).to('cuda')
                output = model(sample)
                latent_outputs.append(output)
                
            save = [latent_outputs]
            torch.save(save,full_path)


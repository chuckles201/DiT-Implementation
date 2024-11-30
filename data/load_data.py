import torchvision
import torch
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from load_latents import load_latents
import glob


'''CelebDataset Pipeline

Will simply download the desired data, size
it to the desired size, and crop it.
    
'''
class CelebData(Dataset):
    def __init__(self,split,im_path,im_size=256,im_channels=3,
                 im_ext='jpg',use_latents=False,latent_path=None):
        
        # defining desired size for our
        # dataset
        # use_latents: what?
        
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        
        
        # load_images DataSet function
        self.images = self.load_images(im_path)
        
        # To save and store latents for later use:
        # implement this soon!
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print(f'Found {len(self.latent_maps)} latents')
            else:
                print("Latents not found.")
             
       
    # load and import images
    def load_images(self,im_path):
        assert os.path.exists(im_path), "Image Path not found"
        
        # for each image with given name,
        # load this in, glob searches
        # os.path ensures path is correct.
        ims = []
        fnames = glob.glob(os.path.join(im_path,'CelebA-HQ-img/*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(im_path,'CelebA-HQ-img/*.{}'.format('png')))
        fnames += glob.glob(os.path.join(im_path,'CelebA-HQ-img/*.{}'.format('jpeg')))
        for fname in tqdm(fnames):
            ims.append(fname)
            
        print("found images")
        
        return ims
        
        
                
            
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        if self.use_latents:
            # maps image to latent
            latent = self.latent_maps[self.images[index]]
            return latent
        
        else:
            # actually opening the image and
            # passing it through the VAE,
            # done during training
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor()
            ])(im) # applying to im
            im.close()
            
            # converting output to 
            # appropriate range
            im_tensor = (2*im_tensor) - 1
            return im_tensor
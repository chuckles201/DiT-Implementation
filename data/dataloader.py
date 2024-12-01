import PIL.Image
import torchvision
import torch
import os
from tqdm import tqdm
import glob
import pandas as pd

import PIL

# for reading our image.
from torchvision.io import read_image


'''GetRawImages
Here, we will load our raw images
from the dataset, so we can eventually 
make our latent-mappings.

Only loading all if specified
'''
def raw_images_convert(dataset,dir,extension='jpg',num='all'):
    if os.path.exists(dir):
        print("Path exists, process skipped")
    else:
        os.makedirs(dir,exist_ok=True)
        
        # save all images
        if num == 'all':
            t = tqdm(enumerate(dataset))
            for idx, (image,label) in t:
                image_path = os.path.join(dir,f"{idx}.{extension}")
                image.save(image_path)
                
            print("Saved all ", len(dataset)," images in ", dir)
            
        else:
            t = tqdm(enumerate(dataset))
            final_num = 0
            for idx, (image,label) in t:
                if idx < num:
                    image_path = os.path.join(dir,f"{idx}.{extension}")
                    image.save(image_path)
                    final_num=idx
                    
            print("Saved ", final_num," images in ", dir)



'''DataLoader

Here, we will be creating a custom class that
works with TorchVision, and is able to both load
images of desired shape/size, and replace them with 
latent representations, if avaialable.

The functions we need to implement in order for
it to be compatible is:
1. Init
2. __len__
3. get_item()

Our class will create a file for the images,
and a file for the annotations.
'''

from torch.utils.data.dataset import Dataset

class ImageDataset(Dataset):
    def __init__(self,
                 transform,
                 im_path,
                 im_extension='jpg',
                 use_latents = False,
                 latent_folder = None):
        
        # parameters for loading our 
        # images
        self.use_latents = use_latents
        self.latent_folder = latent_folder
        self.im_path = im_path
        self.im_extension = im_extension
        self.transform = transform
        
        self.load_images() # for imgs.
        
    # loading all images, required
    # for length function
    def load_images(self):
        # search all with given extension
        fnames = glob.glob(os.path.join(self.im_path,f"*.{self.im_extension}"))
        self.images = fnames
        
        
    # first required function
    def __len__(self):
        return len(self.images)
        
        
    def __getitem__(self, index):
        # index the given item,
        # then do transform
        if not self.use_latents:
            im = PIL.Image.open(os.path.join(self.im_path,f'{index}.{self.im_extension}'))
            image = self.transform(im)
            im.close()
        else:
            # typical latent-name
            full_path = os.path.join(self.latent_folder,'latent_storage.pt')
            imgs_file = torch.load(full_path)
            print(imgs_file)
            # loading from correct index,
            # need [0] for base-list
            image = imgs_file[0][index]
            
            # index should match what was loaded!
            
        return image
    





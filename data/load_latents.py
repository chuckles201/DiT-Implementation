import pickle
import glob
import os
import torch


'''Load_latents
Simply assigns a dictionary
of images-> latents,
so we don't have to run the VAE 
encoder every single time.

If the latents are spread across multiple 
files, we may need to load them from multiple
directories

The latents should each refrence an iamge 
'''

# NEED TO DEFINE THIS FUNCTION
# FOR SAVING LATENTS SO WE CAN
# SEE HOW THIS WORKS
def load_latents(path):
    latent_maps = {}
    
    # iterate over all paths in directory
    # with .pkl
    for fname in glob.glob(os.path.join(path,'*.pkl')):
        # opening with pickle in binary
        # read mode
        s = pickle.load(open(fname,'rb'))
        
        # loading file into dictionary
        for k,v in s.items():
            # 0th bc in list
            latent_maps[k] = v[0]
            
    return latent_maps
    
    
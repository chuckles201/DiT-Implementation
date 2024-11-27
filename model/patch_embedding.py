import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

# making sure we have the correct device
device = 'cuda'
device = torch.device(device)



''' Position embedding

- takes in the position embedding dimensionality,
and the grid height and width, and the hidden-size for
the dimensionality of embeddings
- For each x, and each y, will produce a position embedding,
producing one pair for each element in the grid.
- Sinusodial embeddings

'''
def get_patch_pos_embed(dim, grid_dim,device):
    grid_size_h,grid_size_w = grid_dim
    
    # creating 'grid' for our embeddings
    grid_h = torch.arange(grid_size_h,dtype=torch.float32,device=device)
    grid_w = torch.arange(grid_size_w,dtype=torch.float32)


    # embedding = 10k ^ (2i/d_model)

'''Patch Embedding
In this part of our model, we are simply taking 2d patches
of our latent-image, and converting them into D-dimensional
tokens that will attend to eachother in the later transformer
layers.

We will learn one linear layer for each patch.

'''
class PatchEmbedding(nn.Module):
    # Takes input images, and outputs
    #\ patches of desired h and w and c
    # then, we add the position embeddings
    #\ according to x/y chors
    
    def __init__(self,
                 height,
                 width,
                 channels,
                 patch_height,
                 patch_width,
                 hidden_size):
        super().__init__()
        
        self.image_height = height
        self.image_width = width
        self.im_channels = channels
        
        self.patch_height = patch_height
        self.patch_width = patch_width
        
        # inputing h*w*c
        input_dim = channels*height*width
        # passing through layer to change
        # dimensionality
        self.patch_embed = nn.Sequential(
            nn.Linear(input_dim,hidden_size)
        )
        
        ###########################
        # Initializes weights based on the number
        #\ of inputs, and outputs for the layer.
        nn.init.xavier_uniform_(self.patch_embed)
        # initializing only nn.sequential part to start
        # out with a zero bias
        nn.init.constant_(self.patch_embed[0].bias,0)
        
    def forward(self,x):
        grid_size_h = self.image_height // self.patch_height
        grid_size_w = self.image_width // self.patch_width
         
        # (B,C,H,W) -> (B,P,Ph*Pw,D)
        # advance based on dimensionality; axes permuted
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nw nh) (ph pw c)',
                        nh = grid_size_h,
                        nw = grid_size_w)
        
        out = self.patch_embed(out)
        
        # getting concated x/y pos_emb (flattened)
        pos_embed = get_patch_pos_embed()
        
        out += pos_embed
        return out
    
        
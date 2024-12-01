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


Furthermore, the sinusodial embeddings effectively ensure that 
for each vector, the embeddings will vary over time with different 
wavelengths, allowing a variety of things to be expressed with 
various frequencies.

'''
def get_patch_pos_embed(dim, grid_dim,device):
    grid_size_h,grid_size_w = grid_dim
    
    # creating 'grid' for our embeddings
    grid_h = torch.arange(grid_size_h,dtype=torch.float32,device=device)
    grid_w = torch.arange(grid_size_w,dtype=torch.float32)
    
    # getting positions
    grid = torch.meshgrid([grid_h,grid_w],indexing='ij')
    grid_h_pos = grid[0]
    grid_w_pos = grid[1]

    # emb = 10k ^ (2i/d_model)
    # getting the weight/factor term [0,1]
    factor = 10000 ** (torch.arange(start = 0,
                                   end = dim // 4,
                                   dtype=torch.float32,
                                   device=device) / (dim // 4))
    
    # multiplying across each i, for each timestep
    # having dim/4 for each sin and cos, which will
    # be concated
    emb_h = grid_h_pos[:,None].repeat(1,dim//4) / factor
    emb_h = torch.cat([torch.sin(emb_h),torch.cos(emb_h)],dim=-1)
    
    emb_w = grid_w_pos[:,None].repeat(1,dim//4) / factor
    emb_w = torch.cat([torch.sin(emb_w),torch.cos(emb_w)],dim=-1)
    
    pos_emb = torch.cat([emb_h,emb_w],dim=-1)
    
    return pos_emb

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
        self.hidden_size = hidden_size
        
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
        # for each patch, maintain dim size D
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nw nh) (ph pw c)',
                        nh = grid_size_h,
                        nw = grid_size_w)
        
        out = self.patch_embed(out)
        
        # getting concated x/y pos_emb (flattened)
        pos_embed = get_patch_pos_embed(dim=self.hidden_size, grid_dim=[grid_size_h,grid_size_w])
        
        out += pos_embed
        
        return out
    
        
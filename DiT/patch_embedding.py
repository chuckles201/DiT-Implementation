import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import yaml
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
    
    def __init__(self,config):
        super().__init__()
        
        self.image_height = config['im_height']
        self.image_width = config['im_width']
        self.im_channels = config['im_channels']
        self.hidden_size = config['h_dim']
        
        self.patch_height = config['patch_height']
        self.patch_width = config['patch_width']
        
        # inputing h*w*c
        input_dim = self.im_channels*self.patch_height*self.patch_width
        # passing through layer to change
        # dimensionality
        self.patch_embed = nn.Sequential(
            nn.Linear(input_dim,self.hidden_size)
        )
        
        ###########################

        nn.init.xavier_uniform_(self.patch_embed[0].weight)

        nn.init.constant_(self.patch_embed[0].bias,0)
        
    def forward(self,x):
        # create patches shape
        # B,P,H_dim
        
        # num patches h/w
        grid_height = self.image_height // self.patch_height
        grid_width = self.image_width // self.patch_width
        
        # B,C,H,W -> B,H,W,C
        x = x.permute(0,2,3,1).contiguous()
        # B,nh*ph, nw*pw, C -> splitting dims
        x = x.view(x.shape[0],grid_height,self.patch_height,grid_width,self.patch_width,x.shape[3])
        # B,nh,ph,nw,pw,C -> B,nh*nw,ph*pw*c
        x = x.permute(0,1,3,2,4,5).contiguous()
        x = x.view(x.shape[0],x.shape[1]*x.shape[2],x.shape[3]*x.shape[4]*x.shape[5])
        
        x = self.patch_embed(x)
        
        return x
    
        
# # testing-out
# with open('./config/config.yaml','r') as file:
#     config = yaml.safe_load(file)
    
# print(config['dit_params']['im_height'])
# pe = PatchEmbedding(config['dit_params'])

# # should embed this to:
# # B,num_patches=128*128, 768
# t = torch.randn([32,3,256,256])
# p = pe(t)
# print(p.shape)

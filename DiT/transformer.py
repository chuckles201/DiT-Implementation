import torch.nn as nn
from torch.nn import functional as F 
from patch_embedding import PatchEmbedding
from transformer_layer import TransformerBlock
import torch
from einops import rearrange

'''Time Embedding
Our time embedding, which we determine the parameters of,
and is added along with the context/class description to
control the LN/Scale parameters in our Transformer Layer.

A sinusodial embedding done the same as our position emb.
'''
def get_time_embedding(input,dim):
    assert(dim % 2 == 0)
    # takes in shape (B,1)
    # output B,t_emb
    factor = 10000**(torch.arange(0,dim // 2,
                                  dtype=torch.float32,
                                  device=input.device) / (dim // 2))
    # want: B,Emb
    # EXTRE UNSQUEEZE
    out_emb = input[:,None].repeat(1,dim // 2) / factor.unsqueeze(0)
    out_emb = torch.cat([torch.sin(out_emb),torch.cos(out_emb)],dim=-1)
    return out_emb

# # test
# t = torch.randint(0,18,size=(12,))
# get_time_embedding(t,)

'''DiT

Our DiT model, now just puts
together
1. Time-embedding assignment
2. PAtchify/de-patchify
3. All transformer blocks

We encode the timestep
the same as the hidden-dim
'''

class DiT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        # starting 256 hidden-size?
        self.patchify = PatchEmbedding(config)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config['layers'])]
        )
        # projection for final de-patch
        self.norm = nn.LayerNorm(config['h_dim'])
        self.out_proj = nn.Linear(config['h_dim'],config['im_channels']*config['patch_height']*config['patch_width'])
        
        # adaln
        # double-proj
        self.adaln = nn.Sequential(
            nn.Linear(config['timestep_dim'],config['h_dim']),
            nn.SiLU(),
            nn.Linear(config['h_dim'],config['h_dim']),
            nn.SiLU(),
            nn.Linear(config['h_dim'],config['h_dim']*2)
        )
        
        ### Weight init to start proj at zero ###
        # starting out to be zero
        nn.init.constant_(self.out_proj.weight,0)
        nn.init.constant_(self.out_proj.bias,0)
        
        # starting out adaln to be zero
        nn.init.constant_(self.adaln[-1].weight,0)
        nn.init.constant_(self.adaln[-1].bias,0)
        ### IGNORING OTHER INITs
        
    def forward(self,x,t):
        # B,t_emb
        time_emb = get_time_embedding(t,self.config['timestep_dim'])
        # adaln params
        # B,1,emb
        adaln = self.adaln(time_emb).unsqueeze(-2).chunk(2,dim=-1)
        
        # passing all thru!
        for layer in self.transformer_layers:
            x = layer(x,time_emb)
            
        x = self.norm(x) * (adaln[0] + 1) + adaln[1]
        x = self.out_proj(x)
        
        # defining shapes
        b,h,w,c = x.shape[0],self.config['im_height'],self.config['im_width'],self.config['im_channels']
        num_height = self.config['im_height']//self.config['patch_height']
        num_width = self.config['im_height']//self.config['patch_height']
        
        
        # b,p,emb -> b,c,h,w 
        # each patch has 2x2 pixels
        # which we'll stick together
        de_patch = rearrange(x, 'b (nh nw) (ph pw c) -> b (nh ph) (nw pw) c',
                             c = c,
                             nh = num_height,
                             nw = num_width,
                             ph = h//num_height,
                             pw = w//num_width,
                             b = b)
        
        return de_patch

        
        



# import yaml   
# with open('./config/config.yaml') as file:
#     config = yaml.safe_load(file)
# # testing:
# t = torch.randn([8,16*16,768])
# time = torch.randint(0,500,size=(8,))
# model = DiT(config['dit_params'])

# # B,p,hidden
# print(sum([p.numel() for p in model.parameters()]))

# print(model(t,time).shape)
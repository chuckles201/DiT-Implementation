import torch
from torch import nn
import torch.nn.functional as F 


from attention import Attention




'''Transformer Layer

Our transformer layer will be a classic
pre-ln transformer, with the addition of
parameters for normalization, that takes
in the class and time-embeddings.

takes:
-numheads,
-d_hidden

Our AdaLN parameters
will start at zero, so our
block is intially an identity block.

We init weights with xavier and biases
with zero. 

- Xavier factors in nout, which 
is needed for stability during backward pass.
Xavier also works well with layernorm, because
xavier assumes a subsequent activation.

'''

class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_heads = config['num_heads']
        self.h_dim = config['h_dim']
        self.d_head = self.h_dim //self.num_heads
        self.attn_norm = nn.LayerNorm(self.h_dim)
        
        self.timestep_dim = config['timestep_dim']
        
        # takes B,P,Emb
        self.attn = Attention(config)
        
        # projection for adaln params
        # 6 for each transformer-block
        # DIFFERS FROM PAPER
        self.t_proj = nn.Sequential(
            nn.Linear(self.timestep_dim,self.h_dim),
            nn.SiLU(),
            nn.Linear(self.h_dim,self.h_dim),
            nn.SiLU(),
            nn.Linear(self.h_dim,6*self.h_dim)
        )

        
        # mlp layer of transformer
        self.mlp_1 = nn.Linear(self.h_dim,self.h_dim*4)
        self.mlp_2 = nn.Linear(self.h_dim*4,self.h_dim)
        self.mlp_norm = nn.LayerNorm(self.h_dim)
        self.silu = nn.SiLU()
        
        # initializes weights at zero
        # for parameters
        nn.init.constant_(self.t_proj[0].weight,0)
        nn.init.constant_(self.t_proj[0].bias,0)
        nn.init.constant_(self.t_proj[2].weight,0)
        nn.init.constant_(self.t_proj[2].bias,0)
        nn.init.constant_(self.t_proj[4].weight,0)
        nn.init.constant_(self.t_proj[4].bias,0)
        
        
        ### Other weight-inits! ###
        nn.init.xavier_uniform_(self.mlp_1.weight)
        nn.init.xavier_uniform_(self.mlp_2.weight)
        nn.init.constant_(self.mlp_1.bias,0)
        nn.init.constant_(self.mlp_2.bias,0)
        
    # takes in x, t 
    # TODO: add class functionality!
    def forward(self,x,time):
        # scale/shift1,alpha1,scale/shift2,alpha2
        # unsqueeze so alligns w/ each
        ada_ln_params = self.t_proj(time).unsqueeze(-2).chunk(6,dim=-1)
        # attn layer
        residue = x
        x = self.attn_norm(x) * (ada_ln_params[0]+1) + ada_ln_params[1]
        x = self.attn(x)
        x = (x*ada_ln_params[2]) + residue
        
        # mlp layer
        residue = x
        x = (self.mlp_norm(x) * (ada_ln_params[3]+1)) + ada_ln_params[4]
        x = self.mlp_1(x)
        x = self.silu(x)
        x = self.mlp_2(x)
        x = (x*ada_ln_params[5]+residue)
        
        return x
    
    
### all works###

# import yaml   
# with open('./config/config.yaml') as file:
#     config = yaml.safe_load(file)
# # testin:
# t = torch.randn([32,16*16,768])
# time = torch.randn([32,768])
# transf = TransformerBlock(config['dit_params'])

# # B,p,hidden
# print(transf(t,time).shape)
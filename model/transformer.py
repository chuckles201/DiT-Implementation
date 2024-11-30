import torch.nn as nn
from torch.nn import functional as F 
from patch_embedding import PatchEmbedding
from transformer_layer import TransformerLayer
from einops import rearrange
import torch

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


'''Transformer Block
Our responsiblity our transformer, is to patchify
our image, and assign our time and class embedding, 
so that we can pass it through our network of
transformer layers, and at the end re-arange our output
into a latent-image for our training-objective.

There will be a layernorm followed by the patchification.

The sinusodial time embedding will be projected with a 
small MLP.

Our 'de-patchify block' will have scale and shift parameters
controlled by class and time embed, with a final linear
projection. (and then re-matching the patches with shapes)


'''

class DiT(nn.Module):
    def __init__(self,im_size,im_channels,config):
        super.__init__()
        # take in arguments from yaml file
        # patch_size: 2
        # num_layers: 12
        # hidden_size: 768
        # num_heads: 12
        # head_dim: 64
        # timestep_embd_dim: 768 
        
        # patch embedding takes:
        # def __init__(self,
        #          height,
        #          width,
        #          channels,
        #          patch_height,
        #          patch_width,
        #          hidden_size):
        self.image_height = im_size # SAME H,W
        self.image_width = im_size  
        self.im_channels = im_channels      
        self.hidden_size = config['hidden_size']
        self.patch_height = config['patch_size']
        self.patch_width = config['patch_size']
        
        self.nh = self.image_height // self.patch_height
        self.nw = self.image_width // self.patch_width
        
        self.timestep_emb_dim = config['timestep_emb_dim']
        
        # middle transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config['num_layers'])
        ])
        
        
        # patch embedding block encodes to hidden
        # size with inferred amount of patches
        self.patch_embed_layer = PatchEmbedding(self.image_height,
                                                self.image_width,
                                                self.im_channels,
                                                self.patch_height,
                                                self.patch_width,
                                                self.hidden_size)
        
        
        # sinusodial time embedding projection (learned)
        self.t_proj = nn.Sequential(
            nn.Linear(self.timestep_emb_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size,self.hidden_size)
        )
        
        
        # de-patching block will be layernorm and Linear
        # layernorm is *not* learned by itself.
        self.norm = nn.LayerNorm(self.hidden_size,elementwise_affline=False,eps=1e-6)
        # (B, P, Hidden) -> (B, P*h*w*c)
        self.final_proj = nn.Linear(self.hidden_size,self.patch_height*self.patch_width*self.im_channels)
        # parameters for final layernorm
        self.ada_ln_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size,2*self.hidden_size,bias=True)
        )
        
        # Initializing (Xavier is chosen)
        # playing with std?? -> bc normal
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)
        
        # start final weights and bias at zero
        nn.init.constant_(self.final_proj.weight,0)
        nn.init.constant_(self.final_proj.bias,0)
        
        # start at zero, slowly learn
        nn.init.constant_(self.ada_ln_final[1].weight,0)
        nn.init.constant_(self.ada_ln_final[1].bias,0)
        
    # where is our class embedding?
    def forward(self,x,t):
        
        # our tokens to pass through
        tokens = self.patch_embed_layer(x)
        
        # our time embedding
        time_emb = get_time_embedding(t)
        time_emb = self.t_proj(time_emb)
        
        # passing through transformer blocks
        output = tokens
        for layer in self.layers:
            # unchanged time embedding
            output = layer(output,time_emb)
            
        # layernorm final
        mlp_shift,mlp_scale = self.ada_ln_final(time_emb).chunk(2,dim=-1)
        output = self.norm(output) * (mlp_shift.unsqueeze(1)+1) + mlp_scale.unsqueeze(1)
        
        # final projection
        # (B, P, Emb) -> (B, P, ph*pw*C)
        output = self.final_proj(output)
        # (B, P=(nh*nw), ph*pw*C) -> (B, C, (nh ph), (nw pw))
        # shapes will auto-multiply for hw/nw
        # sharing dims is like normal, but
        # actual boundaries are hidden
        output = rearrange(output,'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                           ph = self.patch_height,
                           pw = self.patch_width,
                           c = self.im_channels,
                           nh = self.nh,
                           nw = self.nw)
        
        return output # final noise-predictions
        
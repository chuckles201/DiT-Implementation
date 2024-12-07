import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

''' Simple multi head self-attention,
the backbone of the DiT.

Remember, each patch looks at the embeddings
of each other patch to decide how it should
add the other values to itself.

No mask, because we are not doing
a prediction task where masking would be 
necessary
'''

class Attention(nn.Module):
    # config arguments:
    # patch_size: 2
    # num_layers: 12
    # hidden_size: 768
    # num_heads: 12
    # head_dim: 64
    # timestep_embd_dim: 768 
    def __init__(self, config):
        
        super().__init__()
        self.num_heads = config['num_heads']
        self.attn_dim = config['h_dim']
        self.head_dim = self.attn_dim // self.num_heads
        
        

        
        self.proj = nn.Linear(self.attn_dim,self.attn_dim*3,bias=True)
        self.output_proj = nn.Sequential(
            nn.Linear(self.attn_dim,config['h_dim'])
        )
        
        
        # initializing our weights:
        # more xavier and 0 bias init
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias,0)
        nn.init.xavier_uniform_(self.output_proj[0].weight)
        nn.init.constant_(self.output_proj[0].bias,0)
    def forward(self,x):
        # B, P, Dim
        input_shape = x.shape
        # B, n_heads, P, d_head
        interim_shape = [input_shape[0],input_shape[1],self.num_heads,self.head_dim]
        q,k,v = self.proj(x).chunk(3,dim=-1)
        # B, P, d_head * n_head -> B,n_head,P,d_head
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        
        w = q @ k.transpose(-1,-2)
        w /= (q.shape[-1] ** 0.5)
        w = F.softmax(w,dim=-1)
        out = w @ v
        
        # B, N_head, P, d_head -> B,P,Emb
        out = out.transpose(1,2).contiguous()
        out = out.view(input_shape)
        out = self.output_proj(out)
        return out
import torch
from torch import nn
import torch.nn.functional as F 


from attention import Attention





'''Transformer Block
This is a classic transformer block with pre-LN architecture,
and residual connecctions.

However, we use adaptiveLN-zero, which basically means that our
context (Class we specify), and our denoising timestep will
actually determine the scale and shift parameters for our
layernorm! Furthermore, they'll determine the scale of the 
output of our Attention and our feed-forward network.

This allows us to determine 
1. How much certain features are important/certain after
our layernorm (w/ residual connections)
2. It also allows us to control the relative importance
before the residual connections, or how 'dominant' our output
from FFWD and MHA layer should be relative to the residual
connection.

This was found to be expirimentally the best way to introduce
class context and timestep embeddings into the model.

Alpha's are intialized as zero, so that the network begins as
an identity block, and slowly learns the importance of the ffwd
and MHA blocks (relative to residual connections). The scale
and shift parameters are zero and one (normalized).
'''

class TransformerLayer(nn.Module):
    def __init__(self, config):
        # take in arguments from yaml file
        # patch_size: 2
        # num_layers: 12
        # hidden_size: 768
        # num_heads: 12
        # head_dim: 64
        # timestep_embd_dim: 768 
        super().__init__()
        self.hidden_size = config["hidden_size"]
        
        ff_hidden_dim = 4*self.hidden_size
        
        # B, P, Hidden
        # elementwise affline means *don't* learn scale
        # and shift params during training (we'll teach it!)
        # eps is to add to variance to smoothe-out similarity in outputs.
        self.ln_1 = nn.LayerNorm(self.hidden_size,elementwise_affine=False, eps=1E-6) 
        self.ln_2 = nn.LayerNorm(self.hidden_size,elementwise_affine=False, eps=1E-6) 
        
        # attn
        self.attn = Attention('Hyper params here***')
        
        # 4x dim in linear layer
        self.mlp_block = nn.Sequential(
            nn.Linear(self.hidden_size,ff_hidden_dim),
            nn.GELU(approximate='tanh'), # use GELU with tanh approx.
            nn.Linear(ff_hidden_dim,self.hidden_size)
        )
        
        
        # AdaLN-Zero parameters
        # 1. Convert dimensionality to 6x for each param-vector
        # 2. For each neuron's output, we have:
        # - Layernorm scale and shift intialized at 0,1
        # - MHA and FFWD output scale init at 0
        # 3. The time and class are combined (added)
        
        # these will scale and add to layernorm
        self.adaln_block = nn.Sequential(
            nn.SiLU(), # used in practice
            nn.Linear(self.hidden_size, self.hidden_size * 6, bias=True)
        )
        
        
        ######## WHY? #############
        """
        Explain Xavier init:
        - fld,fals,d
        """
        nn.init.xavier_uniform_(self.mlp_block[0].weight)
        nn.init.constant_(self.mlp_block[0].bias,0)
        nn.init.xavier_uniform_(self.mlp_block[-1])
        nn.init.constant_(self.mlp_block[-1].bias,0)
        
        # initialize layernorm params to zero
        # they are not yet ready to scale/move 
        # the output, until we learn them
        nn.init.constant_(self.adaln_block[-1].weight,0)
        nn.init.constant_(self.adaln_block[-1].bias,0)
        
    def forward(self,x,condition):
        # x is input tensor B,P,Dim
        # condition is added time and class
        #\ B, Dim*6 -> B, 6, dim
        # unsqueeze to expand across dims-> B,1,Emb
        adaln_params = self.adaln_block(condition).chunk(6,dim=-1)
        residue = x
        x = self.ln_1(x)* (adaln_params[0].unsqueeze(1) + 1) + adaln_params[1].unsqueeze(1)
        x = self.attn(x) * adaln_params[2].unsqueeze(1) + residue
        residue = x
        x = self.ln_2(x) * adaln_params[3].unsqueeze(1) + adaln_params[4].unsqueeze(1)
        x = self.mlp_block(x) * adaln_params[5].unsqueeze(1) + residue
        
        return x
        

        
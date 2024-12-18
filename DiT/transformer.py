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


'''Class-Embedding
Embedding for our classes, different than time embedding
in that no classes are automatically similar.

Just acess an embedding-table of len classes, and then
'''
class YEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['num_classes'],config['h_dim'])
        
    # size (B,) labels
    def forward(self,label):
        return self.embedding(label)
    # returns shape(B,dim_embed)

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
        self.n_classes = config['num_classes']
        self.config=config
        self.class_table = YEmbedding(config)
        # starting 256 hidden-size?
        self.patchify = PatchEmbedding(config)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config['layers'])]
        )
        # projection for final de-patch, noise/var
        # norm doesn't learn!
        self.norm = nn.LayerNorm(config['h_dim'],elementwise_affine=False)
        self.out_proj = nn.Linear(config['h_dim'],config['im_channels']*config['patch_height']*config['patch_width']*2)
        
        # initial projection
        # before individuals are learned!
        self.time_proj = nn.Sequential(
            nn.Linear(config['timestep_dim'],config['h_dim']),
        )
            
        self.class_proj = nn.Sequential(
            nn.Linear(config['timestep_dim'],config['h_dim']),
        )   
        
        self.cond_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config['timestep_dim'],config['h_dim'])
        ) 
        
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config['h_dim'],config['h_dim']),
            nn.SiLU(),
            nn.Linear(config['h_dim'],config['h_dim']*2)
        )
        
        ### Weight init to start proj at zero ###
        # starting out to be zero
        nn.init.constant_(self.out_proj.weight,0)
        nn.init.constant_(self.out_proj.bias,0)
        
        # adaln in-bet
        nn.init.xavier_uniform_(self.adaln[1].weight)
        nn.init.constant_(self.adaln[1].bias,0)
        
        # starting out adaln to be zero
        nn.init.constant_(self.adaln[-1].weight,0)
        nn.init.constant_(self.adaln[-1].bias,0)
        
        
        
        ### Xavier init. for stability
        nn.init.xavier_uniform_(self.cond_embed[-1].weight)
        nn.init.constant_(self.cond_embed[-1].bias,0)
        
        nn.init.xavier_uniform_(self.time_proj[0].weight)
        nn.init.constant_(self.time_proj[0].bias,0)
        
        nn.init.xavier_uniform_(self.class_proj[0].weight)
        nn.init.constant_(self.class_proj[0].bias,0)
        
    def forward(self,x,t,y):
        # B,t_emb=hdim
        time_emb = get_time_embedding(t,self.config['timestep_dim'])
        # class emb add-> B,y_emb=hdim
        class_emb = self.class_table(y)
        # adaln params
        # B,1,emb
        cond = self.time_proj(time_emb)+self.class_proj(class_emb)
        cond_embed = self.cond_embed(cond)
        # final projection of both-added
        adaln = self.adaln(cond_embed).unsqueeze(-2).chunk(2,dim=-1)
        
        x = self.patchify(x)
        
        # passing all thru!
        # give condition-embed!
        for layer in self.transformer_layers:
            x = layer(x,cond_embed)
            
        x = self.norm(x) * (adaln[0] + 1) + adaln[1]
        x = self.out_proj(x)
        
        # defining shapes
        b,h,w,c = x.shape[0],self.config['im_height'],self.config['im_width'],self.config['im_channels']
        num_height = self.config['im_height']//self.config['patch_height']
        num_width = self.config['im_height']//self.config['patch_height']
        
        
        # b,p,emb*2 -> b,2,c,h,w,
        # each patch has 2x2 pixels
        # which we'll stick together
        de_patch = rearrange(x, 'b (nh nw) (ph pw c np) -> b np c (nh ph) (nw pw)',
                             c = c,
                             nh = num_height,
                             nw = num_width,
                             ph = h//num_height,
                             pw = w//num_width,
                             b = b,
                             np= 2)
        
        return de_patch

        
        



# import yaml   
# with open('./config/config.yaml') as file:
#     config = yaml.safe_load(file)
# # testing:
# t = torch.randn([1,4,32,32])
# time = torch.randint(0,500,size=(8,))
# model = DiT(config['dit_params'])
# label=torch.tensor([1])

# # B,p,hidden
# print(sum([p.numel() for p in model.parameters()]))

# print(model(t,time,label).chunk(2,dim=1)[0].shape)
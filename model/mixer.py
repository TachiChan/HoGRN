import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MlpBlock(nn.Module):
    def __init__(self, in_features, mlp_dim, out_features, bias=True):
        super(MlpBlock, self).__init__()
        self.mlp_1 = nn.Linear(in_features, mlp_dim, bias=bias)
        self.mlp_2 = nn.Linear(mlp_dim, out_features, bias=bias)
        
    def forward(self, x):
        y = self.mlp_1(x)
        y = F.gelu(y)
        y = self.mlp_2(y)
        return y

class MixerDrop(nn.Module):
    def __init__(self, num_relations, in_features, tokens_mlp_dim, channels_mlp_dim, rel_mask, chan_drop):
        super(MixerDrop, self).__init__()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)

        self.mlpblock_1 = MlpBlock(num_relations, tokens_mlp_dim, num_relations)
        self.mlpblock_2 = MlpBlock(in_features, channels_mlp_dim, in_features)

        self.num_relations  = num_relations
        self.in_features    = in_features   
        
        self.rel_mask   = rel_mask
        self.chan_drop  = chan_drop  

        self.drop_chan	= torch.nn.Dropout(chan_drop)

    def _mask_rel(self, x):
        idx = torch.randperm(self.num_relations).float()
        num_keep = int(np.floor(self.num_relations * (1-self.rel_mask)))
        mask_idx = torch.stack((torch.zeros(num_keep), idx[:num_keep])).long()
        mask = torch.sparse.FloatTensor(mask_idx, torch.ones(num_keep),[1, self.num_relations]).to(x.device) 
        x = mask.to_dense().T * x
        return x

    def forward(self, x, training):
        # inter-relation learning
        if training and self.rel_mask > 0:
            y = self.norm(x)
            y = self._mask_rel(y)
        else:
            y = self.norm(x)
        y = self.mlpblock_1(y.T) 
        x = x + y.T

        # intra-relation learning
        if training and self.chan_drop > 0:
            y = self.norm(x)
            y = self.drop_chan(y)
        else:
            y = self.norm(x)
        y = self.mlpblock_2(y) 
        return x + y    

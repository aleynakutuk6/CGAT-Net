import torch
from torch import nn

from .feed_forward import PositionWiseFeedForward, FeedForward
from .graph_attention_network import GraphAttentionNetwork
from .normalization import Normalization
from .skip_connection import SkipConnection

class GraphTransformerLayer(nn.Module):

    def __init__(
        self, n_heads: int, embed_dim: int, num_self_attns: int, 
        normalization='batch', dropout=0.1, apply_graph_mask: bool=False):
        super().__init__()

        self.self_attentions = nn.ModuleList()
        for _ in range(num_self_attns):
            self.self_attentions.append(
                SkipConnection(
                    GraphAttentionNetwork(
                        n_heads=n_heads,
                        input_dim=embed_dim,
                        embed_dim=embed_dim,
                        dropout=dropout,
                        apply_graph_mask=apply_graph_mask,
                    )
                )
            )
        
        self.tmp_linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * num_self_attns, embed_dim, bias=True),
            nn.ReLU(),
        )
        self.norm1 = Normalization(embed_dim, normalization)
        
        self.positionwise_ff = SkipConnection(
            PositionWiseFeedForward(
               embed_dim=embed_dim,
               dropout=dropout
            )
        )
        self.norm2 = Normalization(embed_dim, normalization)
        
        
    def forward(self, input: torch.Tensor, masks: torch.Tensor):
        
        hh = []
        for l_idx, layer in enumerate(self.self_attentions):
            sel_mask = None if masks is None else masks[:, l_idx, ...]
            hh.append(layer(input, mask=sel_mask))
        hh = torch.cat(hh, dim=2)
        hh = self.tmp_linear_layer(hh)
        hh = self.norm1(hh)
        hh = self.positionwise_ff(hh)
        hh = self.norm2(hh)
        return hh
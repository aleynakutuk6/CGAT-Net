import math
import torch.nn as nn

class FeedForward(nn.Sequential):

    def __init__(self, in_dim: int, out_dim: int, dropout: float=0.1, bias: bool=True):
        super().__init__(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=bias),
            nn.ReLU()
        )


class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.sub_layers = FeedForward(embed_dim, embed_dim, dropout=dropout, bias=True)
        self.init_parameters()


    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    

    def forward(self, input, mask=None):
        return self.sub_layers(input)
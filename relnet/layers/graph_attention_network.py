import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionNetwork(nn.Module):
    
    def __init__(self, n_heads, input_dim, embed_dim=None, 
                 val_dim=None, key_dim=None, dropout=0.1, 
                 apply_graph_mask: bool=False):
        super().__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.apply_graph_mask = apply_graph_mask

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, q, h=None, mask=None, graph_mask=None):
        """
        Args:
            q: Input queries (batch_size, n_query, input_dim)
            h: Input data (batch_size, graph_size, input_dim)
            mask: Input attention mask (batch_size, n_query, graph_size)
                  or viewable as that (i.e. can be 2 dim if n_query == 1);
                  Mask should contain -inf if attention is not possible 
                  (i.e. mask is a negative adjacency matrix)
        
        Returns: 
            out: Updated data after attention (batch_size, graph_size, input_dim)
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"
        assert mask is None or graph_mask is None

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        dropt1_qflat = self.dropout_1(qflat)
        Q = torch.matmul(dropt1_qflat, self.W_query).view(shp_q)

        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        dropt2_hflat = self.dropout_2(hflat)
        K = torch.matmul(dropt2_hflat, self.W_key).view(shp)

        dropt3_hflat = self.dropout_3(hflat)
        V = torch.matmul(dropt3_hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        
        # print(Q.shape, K.shape, V.shape, self.n_heads)

        # Optionally apply mask to prevent attention
        if mask is not None and not self.apply_graph_mask:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility = compatibility + mask.type_as(compatibility)
        
        attn = F.softmax(compatibility, dim=-1)
        
        if mask is not None and self.apply_graph_mask:
            mask = mask.view(
                1, batch_size, n_query, graph_size).expand_as(attn)
            attn = mask * attn

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        
        return out
import numpy as np
import torch.nn as nn

from .graph_transformer_network import GraphTransformerNetwork

class CGATNet(nn.Module):
    
    def __init__(self, cfg_data):
        
        super().__init__()
        n_layers = cfg_data["n_layers"]
        n_heads = cfg_data["n_heads"] 
        n_attns = cfg_data["n_attns"]
        embed_dim = cfg_data["embed_dim"]
        feedforward_dim = cfg_data["feedforward_dim"]
        normalization = cfg_data["normalization"]
        dropout = cfg_data["dropout"]
        n_classes = cfg_data["n_classes"]
        
        backbone = cfg_data["backbone"]
        pretrained_backbone_path = cfg_data["pretrained_backbone_path"]
        backbone_n_classes = cfg_data["backbone_n_classes"]
        freeze_backbone = cfg_data["freeze_backbone"] if "freeze_backbone" in cfg_data else False
        use_cls_backbone_outs = cfg_data["use_cls_backbone_outs"] if "use_cls_backbone_outs" in cfg_data else False
        apply_graph_mask = cfg_data["apply_graph_mask"] if "apply_graph_mask" in cfg_data else False
        self.use_no_mask = cfg_data["use_no_mask"] if "use_no_mask" in cfg_data else False
        
        self.encoder = GraphTransformerNetwork(
            n_layers, n_heads, n_attns, 
            embed_dim, normalization, dropout, backbone, 
            pretrained_backbone_path, backbone_n_classes, 
            freeze_backbone, use_cls_backbone_outs, 
            apply_graph_mask=apply_graph_mask)
        
        self.mlp_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, feedforward_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim, bias=True),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(feedforward_dim, n_classes, bias=True)
        )

        
    def forward(self, sketch, attn_masks, padding_mask=None, true_seq_length=None):
        """
        Args:
            attention_mask: Masks for attention computation (batch_size, seq_length, seq_length)
                            Attention mask should contain -inf if attention is not possible 
                            (i.e. mask is a negative adjacency matrix)
            padding_mask: Mask indicating padded elements in input (batch_size, seq_length)
                          Padding mask element should be 1 if valid element, 0 if padding
                          (i.e. mask is a boolean multiplicative mask)
            true_seq_length: True sequence lengths for input (batch_size, )
                             Used for computing true mean of node embeddings for graph embedding
        
        Returns:
            logits: Un-normalized logits for class prediction (batch_size, n_classes)
        """
        
        # Embed input sequence
        if self.use_no_mask:
            h_outs = self.encoder(sketch, None)
        else:
            h_outs = self.encoder(sketch, attn_masks)
            
        if self.training:
            h, backbone_preds = h_outs
        else:
            h = h_outs
        
        # Mask out padding embeddings to zero
        if padding_mask is not None:
            masked_h = h * padding_mask.type_as(h)
            
        # Compute logits
        logits = self.mlp_classifier(h)
        
        if self.training:
            return logits, backbone_preds
        else:
            return logits
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from relnet.layers import GraphTransformerLayer

class GraphTransformerNetwork(nn.Module):
    
    def __init__(self, n_layers=6, n_heads=8, n_attns=2,
                 embed_dim=512, normalization='batch', dropout=0.1, 
                 backbone="resnet18", pretrained_backbone_path=None, 
                 backbone_n_classes=None, freeze_backbone: bool=False, 
                 use_cls_backbone_outs: bool=False, apply_graph_mask: bool=False):         
        super().__init__()

        maps = {
            # models, output dims
            "resnet18": [models.resnet18, 512],
            "resnet34": [models.resnet34, 512],
            "resnet50": [models.resnet50, 2048], 
            "resnet152": [models.resnet152, 2048],
            "vgg19": [models.vgg19_bn, 4096],
            "vgg16": [models.vgg16_bn, 4096],
            "inception_v3": [models.inception_v3, 2048],
            "mobilenet_v3_small": [models.mobilenet_v3_small, 1024],
            "mobilenet_v3_large": [models.mobilenet_v3_large, 1280],
            "mobilenet_v2": [models.mobilenet_v2, 1280]
        }
            
        assert backbone.lower() in maps
        Model, backbone_out_dim = maps[backbone.lower()]
        
        if pretrained_backbone_path is not None:
            assert backbone_n_classes is not None
            if "inception" in backbone.lower():
                Backbone = Model(num_classes=backbone_n_classes, aux_logits=False)
            else:
                Backbone = Model(num_classes=backbone_n_classes)
            Backbone.load_state_dict(
                torch.load(pretrained_backbone_path)["model"])
            print("* Loaded backbone weights: " + pretrained_backbone_path)
        else:
            if "inception" in backbone.lower():
                Backbone = Model(pretrained=True, aux_logits=False)
            else:
                Backbone = Model(pretrained=True)
            if backbone_n_classes is None:
                backbone_n_classes = 1000
            
                
        self.use_cls_backbone_outs = use_cls_backbone_outs
        self.freeze_backbone = freeze_backbone
            
        if use_cls_backbone_outs:
            self.backbone = Backbone
            backbone_out_dim = backbone_n_classes
            self.backbone_rest = nn.Identity()
            
        else:
            if "resnet" in backbone.lower():
                backbone_modules = list(Backbone.children())[:-1]
                backbone_rest = list(Backbone.children())[-1]
            elif "vgg" in backbone.lower():
                classifier = list(Backbone.children())[-1][:-2]
                backbone_modules = list(Backbone.children())[:-1] + [classifier]
                backbone_rest = list(Backbone.children())[-1][-1]
            elif "mobilenet" in backbone.lower():
                backbone_modules = list(Backbone.children())[:-1]
                backbone_rest = list(Backbone.children())[-1][-1]
            elif "inception" in backbone.lower():
                backbone_modules = list(Backbone.children())[:-2]
                backbone_rest = list(Backbone.children())[-1]

            if not freeze_backbone:
                self.backbone = nn.Sequential(*backbone_modules)
            else:
                self.backbone = nn.ModuleList(backbone_modules)
                
            if pretrained_backbone_path is not None:
                self.backbone_rest = backbone_rest
            else:
                # this case is imagenet pratrained but its last
                # layer has 1000 classes, which is not equal to the
                # requested backbone_n_classes
                self.backbone_rest = nn.Linear(
                    backbone_out_dim, backbone_n_classes)

        # Linear layer to map to embed dim
        assert backbone_out_dim > 0
        self.fc_objs = nn.Linear(backbone_out_dim, embed_dim) # Output: B x S x Embed_dim
        
        # Transformer blocks
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                n_heads, embed_dim, n_attns, normalization, dropout, 
                apply_graph_mask=apply_graph_mask) for _ in range(n_layers)
        ])

    def forward(self, sketch, attn_masks=None):
        
        b, obj_s, obj_c, obj_h, obj_w = sketch.shape
        h = sketch.view(-1, obj_c, obj_h, obj_w)
        if not self.freeze_backbone:
            h = self.backbone(h)
        
        elif self.use_cls_backbone_outs:
            # backbone is frozen and class outputs are used directly
            self.backbone = self.backbone.eval()
            with torch.no_grad():
                h = self.backbone(h)
        
        else:
            num_children, num_tr_layers = len(self.backbone), 2
            for n in range(num_children):
                if n + 2 < num_children:
                    self.backbone[n].eval()
                    with torch.no_grad():
                        h = self.backbone[n](h)
                else:
                    h = self.backbone[n](h)  

        h = h.view(b, obj_s, -1)
        
        if self.training:
            inter_preds = self.backbone_rest(h)
            
        h = self.fc_objs(h)
        
        # Perform n_layers of Graph Transformer blocks
        for layer in self.transformer_layers:
            h = layer(h, masks=attn_masks)
        
        if self.training:
            return h, inter_preds
        else:
            return h

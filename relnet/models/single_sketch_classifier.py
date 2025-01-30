import torchvision.models as models

def get_single_sketch_model(cfg):
        Model, backbone = None, cfg["backbone"]

        if "resnet" in backbone.lower():
            maps = { 
                # models, output dims
                "resnet18": models.resnet18,
                "resnet34": models.resnet34,
                "resnet50": models.resnet50, 
                "resnet152": models.resnet152,
            }
            assert backbone.lower() in maps
            Model = maps[backbone.lower()](num_classes=cfg["n_classes"])
        
        elif "mobilenet" in backbone.lower():
            maps = {
                "mobilenet_v3_small": models.mobilenet_v3_small,
                "mobilenet_v3_large": models.mobilenet_v3_large,
                "mobilenet_v2": models.mobilenet_v2
                
            }
            assert backbone.lower() in maps
            Model = maps[backbone.lower()](num_classes=cfg["n_classes"])
        
        elif "inception" in backbone.lower():
            maps = {
                "inception_v3": models.inception_v3,
            }
            assert backbone.lower() in maps
            Model = maps[backbone.lower()](num_classes=cfg["n_classes"], aux_logits=False, init_weights=True)
            
        elif "vgg" in backbone.lower():
            maps = {
                "vgg16": models.vgg16_bn,
                "vgg19": models.vgg19_bn,
            }
            assert backbone.lower() in maps
            Model = maps[backbone.lower()](num_classes=cfg["n_classes"])
        
        else:
            raise ValueError("Given model is not available: " + backbone)
        
        return Model
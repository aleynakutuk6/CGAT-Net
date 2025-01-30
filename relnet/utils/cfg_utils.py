import json

def read_config(pth):
    if pth is None: return None
    with open(pth, "r") as f:
        data = json.load(f)

    return data


def parse_item(v: str):
    if v.replace(".", "").replace("-", "").replace("e", "").isnumeric():
        if "." in v or "e" in v:
            return float(v)
        else:
            return int(v)
    elif "," in v:
        v_arr = []
        for item in v.split(","):
            v_arr.append(parse_item(item.strip()))
        return v_arr
    elif v == "true":
        return True
    elif v == "false":
        return False
    else:
        return v


def parse_override_params(s: str):
    data = {}
    for part in s.split(";"):
        k, v = part.strip().split("=")
        k = k.strip()
        v = parse_item(v.strip())
        data[k] = v
    return data


def override_params_in_config(cfg, params):
    assert type(cfg) == dict
    for k in cfg:
        if type(cfg[k]) == dict:
            cfg[k] = override_params_in_config(cfg[k], params)
        elif k in params:
            cfg[k] = params[k]
    return cfg
        


def set_nclasses_in_config(config: dict, n_classes: int):
    if "classification" not in config["model"]:
        return config
    
    config["model"]["classification"]["n_classes"] = n_classes
    if "pretrained_backbone_path" in config["model"]["classification"]:
        if config["model"]["classification"]["pretrained_backbone_path"] is None:
            config["model"]["classification"]["backbone_n_classes"] = n_classes
            
    return config


def parse_configs(
    tr_data_path: str=None, 
    model_path: str=None, 
    pipeline_path: str=None, 
    preprocessor_path: str=None,
    val_data_path: str=None,
    override_params: str=None,
    whole_config_path: str=None):
    
    if whole_config_path is not None:
        config = read_config(whole_config_path)
    else:
        config = {
            "dataset": {},
            "model": {},
            "preprocessor": {},
            "pipeline": {}
        }
        
    if tr_data_path is not None:
        config["dataset"] = read_config(tr_data_path)
    if model_path is not None:
        config["model"] = read_config(model_path)
    if pipeline_path is not None:
        config["pipeline"] = read_config(pipeline_path)
    if preprocessor_path is not None:
        config["preprocessor"] = read_config(preprocessor_path)
        
    if val_data_path is not None:
        dcfg = read_config(val_data_path)
        if type(dcfg["val"]) != list:
            config["dataset"]["val"] = [dcfg["val"]]
        else:
            config["dataset"]["val"] = dcfg["val"]
        
        if "test" in dcfg:
            config["dataset"]["test"] = dcfg["test"]
    else:
        if "val" in config["dataset"] and type(config["dataset"]["val"]) != list:
            config["dataset"]["val"] = [config["dataset"]["val"]]
        
    
    if override_params is not None:
        params = parse_override_params(override_params)
        config = override_params_in_config(config, params)     
    
    if "classification" in config["model"]:
        assert "classification" in config["preprocessor"]
        if "out_sketch_size" in config["model"]["classification"]:
            config["preprocessor"]["classification"]["out_sketch_size"] = config["model"]["classification"]["out_sketch_size"]
        if config["preprocessor"]["classification"]["attn_maps"] is None:
            config["preprocessor"]["classification"]["attn_maps"] = []
        config["model"]["classification"]["attn_maps"] = config["preprocessor"]["classification"]["attn_maps"]
        config["model"]["classification"]["n_attns"] = len(config["model"]["classification"]["attn_maps"])
        
        if "pretrained_backbone_path" not in config["model"]["classification"]:
            config["model"]["classification"]["pretrained_backbone_path"] = None
            
        if "model_path" not in config["model"]["classification"]:
            config["model"]["classification"]["model_path"] = None
        
    
    if "segmentation" in config["model"]:
        assert "segmentation" in config["preprocessor"]

    print("Config:", json.dumps(config, indent=4))
    return config
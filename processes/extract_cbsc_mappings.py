import json


with open("../run_results/coco_all_data/labels_info.json", "r") as f:
    coco_classes = json.load(f)["label_to_idx"]
    
data_dir = "../datasets/mapping_files/cbsc_to_coco_mappings.json"
    
with open(data_dir) as clrf:
    cbsc_to_coco = json.load(clrf)
        
for class_name, mapped_class in cbsc_to_coco.items():
    if class_name in coco_classes:
        cbsc_to_coco[class_name] = class_name
                
with open(data_dir.replace(".json", "_updated.json"), 'w') as outfile:
    json.dump(cbsc_to_coco, outfile)
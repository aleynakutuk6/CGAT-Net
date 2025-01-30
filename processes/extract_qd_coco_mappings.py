import json


all_coco_classes = []
with open("coco_classes.txt", "r") as f:
    lines = f.readlines()
for line in lines:
    all_coco_classes.append(line.replace("\n",""))
    
data_dir = "quickdraw_to_coco_v2.json"
    
with open(data_dir) as clrf:
    sketch_to_coco = json.load(clrf)
        
coco_to_sketch, sketch_to_coco_clean = {}, {}
for class_name, mapped in sketch_to_coco.items():
    if mapped is not None:
        coco_to_sketch[mapped] = coco_to_sketch.get(mapped, []) + [class_name]
        sketch_to_coco_clean[class_name] = mapped
coco_classes = list(set(sketch_to_coco_clean.values()))
coco_classes_to_idx = {c: i for i, c in enumerate(coco_classes)}
qd_classes_to_idx = {c.lower(): i for i, c in enumerate(sketch_to_coco.keys())}
        
data = {"coco_to_sketch": coco_to_sketch, "sketch_to_coco": sketch_to_coco_clean,
        "coco_classes": coco_classes, "qd_classes_to_idx": qd_classes_to_idx}
                
with open("coco_qd_mappings.json", 'w') as outfile:
    json.dump(data, outfile)
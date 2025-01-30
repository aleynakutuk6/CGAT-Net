import os
import copy
import time
import argparse

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train', action="store_true", help="Run trainings.")
parser.add_argument("-cl", '--clean_logs', action="store_true", help="Clean the logs.")
parser.add_argument("-csh", '--clean_shs', action="store_true", help="Clean the sh files.")
parser.add_argument("-cr", '--clean_runs', action="store_true", help="Clean the training dir.")
parser.add_argument("-l", '--run_in_long', action="store_true", help="Run in long partition instead of AI.")
args = parser.parse_args()

exps = [
    # expname, lr_sch, tr_data_dir, model_dir, pipeline_dir, override_params
    
    # SINGLE MODEL EXPS
    # ["single_rn18_bw", True, "single_sketch.json", "single/single_resnet18.json", "default_more_iters.json", "learning_rate=1e-4 ; color_images=false"],
    # ["single_rn18_color", True, "single_sketch.json", "single/single_resnet18.json", "default_more_iters.json", "learning_rate=1e-4"],
    # ["single_rn50_color", True, "single_sketch.json", "single/single_resnet50.json", "default_more_iters.json", "learning_rate=1e-4"],
    # ["single_vgg16", True, "single_sketch.json", "single/single_vgg16.json", "default_more_iters.json", "learning_rate=1e-4"],
    # ["single_vgg19", True, "single_sketch.json", "single/single_vgg19.json", "default_more_iters.json", "learning_rate=1e-4"],
    # ["single_mob_v2", True, "single_sketch.json", "single/single_mobilenet_v2.json", "default_more_iters.json", "learning_rate=1e-4"],
    # ["single_mob_v3_small", True, "single_sketch.json", "single/single_mobilenet_v3_small.json", "default_more_iters.json", "learning_rate=1e-4"],
    # ["single_mob_v3_large", True, "single_sketch.json", "single/single_mobilenet_v3_large.json", "default_more_iters.json", "learning_rate=1e-4"],
    # ["single_inception_v3", True, "single_sketch.json", "single/single_inception_v3.json", "default_more_iters.json", "learning_rate=1e-4"],
    # ["single_vgg19_bw", True, "single_sketch.json", "single/single_vgg19.json", "default_more_iters.json", "learning_rate=1e-4 ; color_images=false"],
    # ["single_inception_v3_bw", True, "single_sketch.json", "single/single_inception_v3.json", "default_more_iters.json", "learning_rate=1e-4 ; color_images=false"],
    
    # SINGLE MODEL EXPS WITH FRISS INCLUDED
    # ["single_withfriss_rn18_bw", True, "single_sketch.json", "single/single_resnet18.json", "default_more_iters.json", "learning_rate=1e-4 ; color_images=false ; train_set_list=qd,friss"],
    # ["single_withfriss_rn18_color", True, "single_sketch.json", "single/single_resnet18.json", "default_more_iters.json", "learning_rate=1e-4 ; train_set_list=qd,friss"],
    # ["single_withfriss_rn50_color", True, "single_sketch.json", "single/single_resnet50.json", "default_more_iters.json", "learning_rate=1e-4 ; train_set_list=qd,friss"],
    # ["single_withfriss_vgg16", True, "single_sketch.json", "single/single_vgg16.json", "default_more_iters.json", "learning_rate=1e-4 ; train_set_list=qd,friss"],
    # ["single_withfriss_vgg19", True, "single_sketch.json", "single/single_vgg19.json", "default_more_iters.json", "learning_rate=1e-4 ; train_set_list=qd,friss"],
    # ["single_withfriss_mob_v2", True, "single_sketch.json", "single/single_mobilenet_v2.json", "default_more_iters.json", "learning_rate=1e-4 ; train_set_list=qd,friss"],
    # ["single_withfriss_mob_v3_small", True, "single_sketch.json", "single/single_mobilenet_v3_small.json", "default_more_iters.json", "learning_rate=1e-4 ; train_set_list=qd,friss"],
    # ["single_withfriss_mob_v3_large", True, "single_sketch.json", "single/single_mobilenet_v3_large.json", "default_more_iters.json", "learning_rate=1e-4 ; train_set_list=qd,friss"],
    # ["single_withfriss_inception_v3", True, "single_sketch.json", "single/single_inception_v3.json", "default_more_iters.json", "learning_rate=1e-4 ; train_set_list=qd,friss"],
    
    # SCENE MODEL WITH BACKBONE EXPERIMENTS
    # ["scene_rn18_l2_gr_color", True, "coco.json", "scene/base_resnet18.json", "default.json", "learning_rate=1e-4 ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3,4"],
    # ["scene_incv3_l2_gr_color", True, "coco.json", "scene/base_inception_v3.json", "default.json", "learning_rate=1e-4 ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3,4"],
    # ["scene_vgg19_l2_gr_color", True, "coco.json", "scene/base_vgg19.json", "default.json", "learning_rate=1e-4 ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3,4"],
    
    # SCENE MODEL WITH BACKBONE INTERMEDIATE LOSS EXPERIMENTS
    # ["scene_vgg19_l2_gr_color_backboneloss", True, "coco.json", "scene/base_vgg19.json", "with_backbone_loss.json", "learning_rate=1e-4 ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3,4"],
    
    # SCENE MODEL WITH GRAPH STRUCTURE EXPERIMENTS
    # ["scene_inception_v3_l2_defmask_color", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "learning_rate=1e-4 ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=false ; attn_maps=0,1,2,3,4"],
    # ["scene_inception_v3_l2_nomask_color", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "learning_rate=1e-4 ; n_layers=2 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1,2,3,4"],
    
    # SCENE WITH NUMBER OF LAYERS EXPERIMENTS
    # ["scene_vgg19_l1_gr_color", True, "coco.json", "scene/base_vgg19.json", "default.json", "learning_rate=1e-4 ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3,4"],
    # ["scene_vgg19_l3_gr_color", True, "coco.json", "scene/base_vgg19.json", "default.json", "learning_rate=1e-4 ; n_layers=3 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3,4"],
    # ["scene_vgg19_l4_gr_color", True, "coco.json", "scene/base_vgg19.json", "default.json", "learning_rate=1e-4 ; n_layers=4 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3,4"],
    # ["scene_vgg19_l5_gr_color", True, "coco.json", "scene/base_vgg19.json", "default.json", "learning_rate=1e-4 ; n_layers=5 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3,4"],
    # ["scene_vgg19_l6_gr_color", True, "coco.json", "scene/base_vgg19.json", "default.json", "learning_rate=1e-4 ; n_layers=6 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3,4"],
    
    # SCENE WITH FEED FORWARD DIM RESNET-18 EXPERIMENTS
    # ["scene_rn18_l2_gr_color_ff1024", True, "coco.json", "scene/base_resnet18.json", "with_backbone_loss.json", "learning_rate=1e-4 ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2"],
    # ["scene_rn18_l2_gr_color_ff1024_nosknet", True, "coco.json", "scene/base_resnet18.json", "with_backbone_loss.json", "learning_rate=1e-4 ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2 ; sknet_prob=0.0 ; friss_prob=0.3 ; qd_prob=0.7"],
    # ["scene_rn18_l2_gr_color_ff512", True, "coco.json", "scene/base_resnet18_smaller.json", "with_backbone_loss.json", "learning_rate=1e-4 ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2"],
    # ["scene_rn18_l2_gr_bw_ff1024", True, "coco.json", "scene/base_resnet18.json", "with_backbone_loss.json", "learning_rate=1e-4 ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2 ; color_images=false"],
    
    # NO-MASK MODEL COMPARISONS
    # ["scene_inception_v3_l2_nomask_bw", True, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=2 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # ["scene_vgg19_l2_nomask_bw", True, "coco.json", "scene/base_vgg19.json", "default.json", "color_images=false ; n_layers=2 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # ["scene_resnet18_l2_nomask_bw", True, "coco.json", "scene/base_resnet18.json", "default.json", "color_images=false ; n_layers=2 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # BACKBONE LOSS EFFECT
    # ["scene_inception_v3_l1_nomask_backboneloss0-5_bw", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; backbone_loss_weight=0.5 ; batch_size=16"],
    
    # ["scene_inception_v3_l1_nomask_backboneloss1-0_bw", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=2 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; backbone_loss_weight=1.0 ; batch_size=16"],

    # NUMBER OF LAYERS
    # ["scene_inception_v3_l1_nomask_bw", True, "coco.json", "scene/base_inception_v3.json", "default.json",  "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # ["scene_inception_v3_l3_nomask_bw", True, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=3 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # ["scene_inception_v3_l4_nomask_bw", True, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=4 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # ["scene_inception_v3_l5_nomask_bw", True, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=5 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # BEST MODEL VARIATIONS COMPARISONS
    
    # ["scene_resnet18_gr012_bw_l2_backboneloss0-5", True, "coco.json", "scene/base_resnet18.json", "with_backbone_loss.json", "color_images=false ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2 ; batch_size=16"],
    
    # ["scene_resnet18_gr012_color_l2_backboneloss0-5", True, "coco.json", "scene/base_resnet18.json", "with_backbone_loss.json", "color_images=true ; n_layers=2 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2 ; batch_size=16"],
    
    # ["scene_resnet18_gr012_color_l1_backboneloss0-5", True, "coco.json", "scene/base_resnet18.json", "with_backbone_loss.json", "color_images=true ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2 ; batch_size=16"],
    
    # ["scene_inception_v3_gr012_color_l1_backboneloss0-5", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=true ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2 ; batch_size=16"],
    
    # ["scene_inception_v3_gr012_color_l1_backboneloss0-5_noselfgr", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=true ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2 ; batch_size=16 ; calc_attn_with_self=false"],
    
    # SKNET PRIORITIZE EFFECT
    
    ["scene_inception_v3_l1_nomask_bw_sknetprior", True, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16 ; prioritize_sknet=true ; qd_prob=0.6 ; friss_prob=0.4"],
    
    # COLOR EFFECT 
    
    # ["scene_inception_v3_l1_nomask_color", True, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=true ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # MAX OBJ EFFECT
    
    # ["scene_inception_v3_l1_nomask_bw_maxobjs10", True, "coco_max_objs_10.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # ["scene_inception_v3_l1_nomask_bw_maxobjs20", True, "coco_max_objs_20.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=12"],
    
    # ["scene_inception_v3_l1_nomask_bw_maxobjs5", True, "coco_max_objs_5.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # MAX OBJ EFFECT
    
    # ["scene_inception_v3_l1_nomask_bw_lr1e-3", True, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16 ; learning_rate=1e-3"],
    
    # ["scene_inception_v3_l1_nomask_bw_lr5e-4", True, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16 ; learning_rate=5e-4"],
    
    # ["scene_inception_v3_l1_nomask_bw_lr5e-5", True, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16 ; learning_rate=5e-5"],
    
    # ["scene_inception_v3_l1_nomask_bw_lr1e-5", True, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16 ; learning_rate=1e-5"],
    
    # ["scene_inception_v3_l1_nomask_bw_lr1e-4_nolrsch", False, "coco.json", "scene/base_inception_v3.json", "default.json", "color_images=false ; n_layers=1 ; use_no_mask=true ; apply_graph_mask=false ; attn_maps=0,1 ; batch_size=16"],
    
    # GRAPH EFFECT
    
    # ["scene_inception_v3_gr01234_bw_l1_backboneloss0-5_noselfattn", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3,4 ; batch_size=16 ; calc_attn_with_self=false"],
    
    # ["scene_inception_v3_gr0123_bw_l1_backboneloss0-5_noselfattn", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2,3 ; batch_size=16 ; calc_attn_with_self=false"],
    
    # ["scene_inception_v3_gr013_bw_l1_backboneloss0-5_noselfattn", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,3 ; batch_size=16 ; calc_attn_with_self=false"],
    
    # ["scene_inception_v3_gr012_bw_l1_backboneloss0-5_noselfgr", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2 ; batch_size=16 ; calc_attn_with_self=false"],
    
    # ["scene_inception_v3_gr14_bw_l1_backboneloss0-5_noselfattn", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=1,4 ; batch_size=16 ; calc_attn_with_self=false"],
    
    # ["scene_inception_v3_gr02_bw_l1_backboneloss0-5_noselfattn", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,2 ; batch_size=16 ; calc_attn_with_self=false"],
    
    # ["scene_inception_v3_gr012_bw_l1_backboneloss0-5_noselfattn", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2 ; batch_size=16 ; calc_attn_with_self=false"],
    
    # ["scene_inception_v3_gr03_bw_l1_backboneloss0-5_noselfattn", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,3 ; batch_size=16 ; calc_attn_with_self=false"],
    
    # ["scene_inception_v3_gr024_bw_l1_backboneloss0-5_noselfattn", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,2,4 ; batch_size=16 ; calc_attn_with_self=false"],
    
    # ["scene_inception_v3_gr124_bw_l1_backboneloss0-5_noselfattn", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=1,2,4 ; batch_size=16 ; calc_attn_with_self=false"],
    
    
    ["scene_inception_v3_gr012_bw_l1_backboneloss0-5_noselfattn_randomselect", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=0,1,2 ; batch_size=16 ; calc_attn_with_self=false ; random_class_select=true"],
    
    ["scene_inception_v3_gr124_bw_l1_backboneloss0-5_noselfattn_randomselect", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=1,2,4 ; batch_size=16 ; calc_attn_with_self=false ; random_class_select=true"],
    
    ["scene_inception_v3_gr123_bw_l1_backboneloss0-5_noselfattn_randomselect", True, "coco.json", "scene/base_inception_v3.json", "with_backbone_loss.json", "color_images=false ; n_layers=1 ; use_no_mask=false ; apply_graph_mask=true ; attn_maps=1,2,3 ; batch_size=16 ; calc_attn_with_self=false ; random_class_select=true"],

]

for e_idx in range(len(exps)):
    script = "#!/bin/bash"
    script += "\n# "
    script += "\n# CompecTA (c) 2018"
    script += "\n# "
    script += "\n# You should only work under the /scratch/users/<username> directory."
    script += "\n#"
    script += "\n# -= Resources =-"
    
    # expname, lr_sch, tr_data_dir, model_dir, pipeline_dir, override_params
    expname, lr_sch, tr_data_cfg_dir, model_dir, pipeline_dir, override_params = exps[e_idx]
    model_dir = os.path.join("configs/model", model_dir)
    pipeline_dir = os.path.join("configs/pipeline", pipeline_dir)
    preprocessor_dir = os.path.join("configs/preprocessor", "only_classification.json")
    tr_data_dir = os.path.join("configs/dataset", tr_data_cfg_dir)
        
    # val_data_dir = os.path.join("configs/dataset", "val_all.json")
    val_data_dir = os.path.join("configs/dataset", "val_freehand.json")
    
    gpu = "t4" if "resnet18" in expname else "v100"
    
    print("* Exp -->", expname)

    script += f"\n#SBATCH --job-name={expname}"
    script += f"\n#SBATCH --output=logs/{expname}.log"
    script += "\n#SBATCH --nodes 1 "
    script += "\n#SBATCH --mem=40G "
    script += "\n#SBATCH --ntasks-per-node=2"
    script += "\n#SBATCH --gres=gpu:1"
    
    if args.run_in_long:
        script += "\n#SBATCH --partition long "
        script += "\n#SBATCH --constraint=tesla_v100"
        
    else:
        script += "\n#SBATCH --partition ai "
        script += "\n#SBATCH --account=ai "
        script += "\n#SBATCH --qos=ai "
        script += f"\n#SBATCH --constraint=tesla_{gpu}"
    
    script += "\n#SBATCH --time=120:00:00 "
    script += "\n#SBATCH --mail-type=ALL"
    script += "\n#SBATCH --mail-user=akutuk21@ku.edu.tr"
    script += "\n"
    script += "\n# Please read before you run: http://login.kuacc.ku.edu.tr/#h.3qapvarv2g49"
    script += "\n################################################################################"
    script += "\n##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################"
    script += "\n################################################################################"
    script += "\n"
    script += "\n# Load Anaconda"
    script += "\nmodule unload cuda"
    script += "\nmodule unload cudnn"
    script += "\nmodule add cuda/10.2"
    script += "\nmodule add cudnn/7.6.5/cuda-10.2"
    script += "\nmodule load anaconda/5.2.0 "
    script += "\nmodule add nnpack/latest "
    script += "\nmodule add rclone "
    script += "\n"
    script += "\n# Set stack size to unlimited"
    script += "\nulimit -s unlimited "
    script += "\nulimit -l unlimited "
    script += "\nulimit -a "
    script += "\necho"
    script += "\n"
    script += "\n################################################################################"
    script += "\n##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################"
    script += "\n################################################################################"
    script += "\n"
    script += "\nsource activate pysgg-new"
    script += "\n\n"
    
    train_txt  = f"python train.py -exp {expname} "
    train_txt += f"-tdc {tr_data_dir} -vdc {val_data_dir} "
    train_txt += f"-mc {model_dir} -pc {pipeline_dir} -prc {preprocessor_dir} "
    train_txt += f'-op "{override_params}" '
    if lr_sch:
        train_txt += "--use-lr-scheduler"
        
    script += train_txt
    
    if args.train:
        with open(f"train_{expname}_temp.sh", "w") as f:
            f.write(script)
        time.sleep(0.5)
        os.system(f"sbatch train_{expname}_temp.sh")
    
    else:
        if args.clean_runs:
            os.system(f"rm -rf run_results/{expname}")
        if args.clean_logs:
            os.system(f"rm -f logs/{expname}.log")
            
if args.clean_shs:
    os.system(f"rm -f *_temp.sh")

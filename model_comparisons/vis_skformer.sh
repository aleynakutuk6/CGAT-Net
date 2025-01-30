#!/bin/bash
# 
# CompecTA (c) 2018
# 
# You should only work under the /scratch/users/<username> directory.
#
# -= Resources =-
#SBATCH --job-name=vis_skformer
#SBATCH --output=logs/vis_skformer.log
#SBATCH --nodes 1 
#SBATCH --mem=40G 
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --partition ai 
#SBATCH --account=ai 
#SBATCH --qos=ai 
#SBATCH --constraint=tesla_t4
#SBATCH --time=120:00:00 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akutuk21@ku.edu.tr

# Please read before you run: http://login.kuacc.ku.edu.tr/#h.3qapvarv2g49
################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

# Load Anaconda
module unload cuda
module unload cudnn
module add cuda/10.1
module add cudnn/7.6.5/cuda-10.1
module load anaconda/5.2.0 
module add nnpack/latest 
module add rclone 

# Set stack size to unlimited
ulimit -s unlimited 
ulimit -l unlimited 
ulimit -a 
echo

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################

source activate skformer

python3 visualize_classify.py -m sketchformer -d cbsc -c common

python3 visualize_classify.py -m sketchformer -d friss -c common
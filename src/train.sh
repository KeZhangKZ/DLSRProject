#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1

# set the number of tasks (processes) per node.
#SBATCH --ntasks-per-node=1

#SBATCH --partition=vgpu40
#SBATCH --gres=gpu:1

# number of CPU cores per task
#SBATCH --cpus-per-task=8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=k.zhang3@uq.net.au


# Create a Python 3.9 virtual environment in your home directory, then activate it:

# python3.9 -m venv ~/python-3.9-project-vgpu40

source /home/Student/s4427443/python-3.9-project-vgpu40/bin/activate

# # # Upgrade version of pip as it sometimes helps with installing packages

# pip install --upgrade pip

# # # Then install whatever you need with pip, e.g.:


# pip install tqdm
# pip install opencv-python
# pip install numpy
# pip install matplotlib
# pip install imageio
# pip install scikit-image
# pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


#                 --pre_train ../experiment/edsr_baseline_x2/model/model_best.pt \

				
# python3 main.py $1 $2 \
#                 $3 $4 \
#                 $5 $6 \
# 				--model EDSR \
#                 --patch_size 256 \
#                 --dir_data /home/Student/s4427443/edsr1/MRI \
#                 --data_range 1-28157/28158-31677 \
#                 --save edsr_baseline_x2_epoch300_lr \
#                 --test_every 1 \
#                 --batch_size 16 \
#                 --n_colors 1 \
#                 --reset \
#                 --no_augment

python3 main.py $* \
                --patch_size 256 \
                --dir_data /home/Student/s4427443/edsr1/MRI \
                --data_range 1-28157/28158-31677 \
                --test_every 1 \
                --n_colors 1 \
                --reset \
                --no_augment
#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1


# set the number of tasks (processes) per node.
#SBATCH --ntasks-per-node=1

#SBATCH --partition=vgpu

# number of CPU cores per task
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1


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

# echo $1
# echo $2
# echo $3
# echo $4
# echo $5
# echo $6
				
python3 main.py $* \
                --patch_size 256 \
                --dir_data /home/Student/s4427443/edsr40/MRI \
                --data_range 1-28157/28158-31677 \
                --test_every 1 \
                --n_colors 1 \
                --reset \
                --no_augment
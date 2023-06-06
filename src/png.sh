#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1

#SBATCH -o pngpre.out

# set the number of tasks (processes) per node.
#SBATCH --ntasks-per-node=1

#SBATCH --partition=vgpu


# set name of job
#SBATCH --job-name=pngpre

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=k.zhang3@uq.net.au


# source /home/Student/s4427443/python-3.9-project/bin/activate
source /home/Student/s4427443/python-3.9-project-vgpu40/bin/activate

pip install --upgrade pip


pip install SimpleITK


python3 png_helper.py $*

echo "HR"
ls /home/Student/s4427443/edsr40/MRI/DIV2K/DIV2K_train_HR | wc -l
echo "LR x2"
ls /home/Student/s4427443/edsr40/MRI/DIV2K/DIV2K_train_LR_bicubic/X2 | wc -l
echo "LR x4"
ls /home/Student/s4427443/edsr40/MRI/DIV2K/DIV2K_train_LR_bicubic/X4 | wc -l
echo "LR x8"
ls /home/Student/s4427443/edsr40/MRI/DIV2K/DIV2K_train_LR_bicubic/X8 | wc -l
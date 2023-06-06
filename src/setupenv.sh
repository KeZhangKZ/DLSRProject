#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1

#SBATCH -o setupenv.out

# set the number of tasks (processes) per node.
#SBATCH --ntasks-per-node=1

#SBATCH --partition=vgpu40
#SBATCH --gres=gpu:1

# number of CPU cores per task
#SBATCH --cpus-per-task=8

# set name of job
#SBATCH --job-name=et

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=k.zhang3@uq.net.au


# Create a Python 3.9 virtual environment in your home directory, then activate it:

python3.9 -m venv ~/python-3.9-project-vgpu40

source /home/Student/s4427443/python-3.9-project-vgpu40/bin/activate

# # Upgrade version of pip as it sometimes helps with installing packages

pip install --upgrade pip

# # Then install whatever you need with pip, e.g.:

pip install tqdm
pip install opencv-python
pip install numpy
pip install matplotlib
pip install imageio
pip install scikit-image
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1

# set the number of tasks (processes) per node.
#SBATCH --ntasks-per-node=1

# number of CPU cores per task
#SBATCH --cpus-per-task=4

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=k.zhang3@uq.net.au


source /home/Student/s4427443/python-3.9-project-vgpu40/bin/activate
				
python3 bicubic.py $*
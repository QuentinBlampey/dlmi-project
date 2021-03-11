#!/bin/bash

#SBATCH --job-name=dlmi_cv
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment
source activate /gpfs/users/piatc/.conda/envs/gr

# Go to the directory where the job has been submitted 
cd ${SLURM_SUBMIT_DIR}

# Execution
python -u cross_validate.py -c efficientnet -k 4 -e 20 -lr 1e-4
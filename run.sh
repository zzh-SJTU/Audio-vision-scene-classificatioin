#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

module load miniconda3
__conda_setup="$('/dssg/opt/icelake/linux-centos8-icelake/gcc-11.2.0/miniconda3-4.10.3-f5dsmdmzng2ck6a4otduqwosi22kacfl/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate pytorch

# training
python train.py --config_file configs/baseline.yaml

# evaluation
python evaluate.py --experiment_path experiments/baseline

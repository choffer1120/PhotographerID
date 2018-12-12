#!/bin/bash

#SBATCH --partition=MIT-6.S198
#SBATCH --reservation=cnh_ABELSON_class
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00

. /etc/profile.d/modules.sh
module add cuda/8.0
module add cudnn/6.0

#nvidia-smi
python PreProccess_24User.py
python resnetTraining_24User.py
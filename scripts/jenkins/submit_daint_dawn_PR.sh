#!/bin/bash -l
#SBATCH --job-name="dawn_PR"
#SBATCH --output=dawn_PR.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cscsci
#SBATCH --constraint=gpu

export CRAY_CUDA_MPS=1

#TODO generalize this script to be able to run possibly other stuff on compute nodes

export PARALLEL_BUILD_JOBS=24
srun ./dawn_PR.sh


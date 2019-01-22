#!/bin/bash
#SBATCH --job-name=gtclang_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=test.log
#SBATCH --time=00:25:00
#SBATCH -C gpu
#SBATCH --cpus-per-task=1
#SBATCH --account=c14
#SBATCH --partition=normal

echo $LD_LIBRARY_PATH

# set environmental parameters

BUILD_DIR=<BUILD_DIR>
cd $BUILD_DIR
<ENV>
<CMD>


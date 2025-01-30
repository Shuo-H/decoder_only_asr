#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --job-name=speechlm_org
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=7:00:00
#SBATCH --output=%j_%x.log
#SBATCH --error=%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuohan@andrew.cmu.edu

. /projects/bcey/shan1/espnet/tools/activate_python.sh
cd /projects/bcey/shan1/espnet/egs2/tedlium2/speechlm1/delta
bash run_org.sh



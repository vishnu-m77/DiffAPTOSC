#!/bin/bash

#SBATCH --mail-user=vmugunda@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name="gpu-test"
#SBATCH --partition=gpu-k80
#SBATCH --account=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:k80:1      # format --gres=gpu:<gpu-type>:<number>
#SBATCH --output=stdout-%x_%j.log
#SBATCH --error=stderr-%x_%j.log

## other gpu partition: gpu-gtx1080ti
echo "Cuda device: $CUDA_VISIBLE_DEVICES"
echo "======= Start memory test ======="

## Load modules
module load  anaconda3/2023.07.1

## Check if MY_CONDA_ENV has already been  created
MY_CONDA_ENV="gpuenv2"

CHK_ENV=$(conda  env list | grep $MY_CONDA_ENV | awk '{print $1}')

echo "CHK_ENV: $CHK_ENV"
if [ "$CHK_ENV" =  "" ]; then
        ## if MY_CONDA_ENV does not exist
        echo "$MY_CONDA_ENV doesn't exist, create it..."
        conda create --yes  --name $MY_CONDA_ENV python=3.10 numba cudatoolkit -c conda-forge -c nvidia
        conda activate $MY_CONDA_ENV
else
        ## if MY_CONDA_ENV already exist
        echo "MY_CONDA_ENV exists, activate $MY_CONDA_ENV"
        #conda init bash
        conda activate $MY_CONDA_ENV
fi

echo "echo: $(which python3) "
echo ""

# put your installs here
pip3 install -r requirements.txt

srun python3 main.py

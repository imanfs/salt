#!/bin/bash

# Job name
#SBATCH --job-name=salt

# choose the GPU queue
#SBATCH -p LIGHTGPU

# requesting one node
#SBATCH --nodes=1
# Only if you really need it!

# keep environment variables
#SBATCH --export=ALL

# requesting 4 V100 GPU
# (remove the "v100:" if you don't care what GPU)
#SBATCH --gres=gpu:a100:1

# number of cpus per task
# don't use this if you have exclusive access to the node
#SBATCH --cpus-per-task=10

# request enough memory
#SBATCH --mem=40G

# Change log names; %j gives job id, %x gives job name
#SBATCH --output=/home/xucabis2/salt/iman/submit/logs/slurm-%j.%x.out
# optional separate error output file
# #SBATCH --error=/home/xucabis2/salt/salt/submit/logs/slurm-%j.%x.err

# speedup (not sure if this does anything)
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
node=$(hostname)
echo "Running on " $node

if [ "$node" == "compute-gpu-0-0.local" ]; then
    echo "Running on light GPU"
    source /home/xzcapwsl/GroupProject/utils/submit/light_setup.sh
    echo $CUDA_VISIBLE_DEVICES
    # Check if a GPU was available
    if [ "$GPU_AVAILABLE" -eq 0 ]; then
        echo "No GPU is available, exiting."
        exit 1
    fi

    echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES
    # export CUDA_VISIBLE_DEVICES=$dev
    # # speedup trick
    export OMP_NUM_THREADS=1

fi

echo $CUDA_VISIBLE_DEVICES
# print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"

# move to workdir
cd /home/xucabis2/salt/salt/
echo "Moved dir, now in: ${PWD}"

# activate environment
eval "$(/share/apps/anaconda/3-2022.05/bin/conda shell.bash hook)"
conda activate ../conda/envs/salt/
echo "Activated environment ${CONDA_DEFAULT_ENV}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi
# run the training
echo "Running training script..."
salt fit \
    --config ~/salt/salt/configs/MaskFormer_base.yaml \
    -c /home/xucabis2/salt/salt/configs/weighting_AlignedMTL.yaml \
    --trainer.precision bf16-true  --force

#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding
# SBATCH -p 3090-gcondo --gres=gpu:1 --gres-flags=enforce-binding

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 00:05:00
#SBATCH -o with_gpu.out
#SBATCH -e with_gpu.err

# Load CUDA module
module load cuda/12.2.2  gcc/10.2   


nvidia-smi

# Compile CUDA program and run
# nvcc -arch sm_20 vecadd.cu -o vecadd
# nvcc -arch=sm_86 -O2 main.cu
nvcc -O2 experiments.cu 
# nsys profile --stats=true --force-overwrite=true --output=outputs/gpu_report_no_warp ./a.out NO_WARP -o ./outputs/NO_WARP.txt
nsys profile --stats=true --force-overwrite=true --output=outputs/gpu_report_multi_warp ./a.out MULTI_WARP -o ./outputs/MULTI_WARP.txt
# ./a.out MULTI_WARP -o ./outputs/MULTI_WARP.txt
# ./a.out NO_WARP -o ./outputs/NO_WARP.txt

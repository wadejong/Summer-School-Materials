#!/bin/bash
#SBATCH --job-name=example1
#SBATCH --output=sbatch.out
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH -p molssi-40core

source /gpfs/projects/molssi/modules-gnu
export OMP_NUM_THREADS=1

mpirun -n 1 ./example1 > output

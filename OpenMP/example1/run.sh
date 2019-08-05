#!/bin/bash
#SBATCH --job-name=example1
#SBATCH --output=sbatch.out
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1
#SBATCH -p molssi

source /gpfs/projects/molssi/modules-gnu
export OMP_NUM_THREADS=1

mpirun -n 1 ./example1 > output
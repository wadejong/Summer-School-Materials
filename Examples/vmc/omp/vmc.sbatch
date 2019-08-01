#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=40 --time=00:20:00 --job-name=vmcomp -p debug-40core

OMP_NUM_THREADS=1 time ./vmc
OMP_NUM_THREADS=2 time ./vmc
OMP_NUM_THREADS=4 time ./vmc
OMP_NUM_THREADS=8 time ./vmc

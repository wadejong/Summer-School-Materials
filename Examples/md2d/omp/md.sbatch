#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=40 --time=00:20:00 --job-name=mdomp -p debug-40core

OMP_NUM_THREADS=32 time ./md
OMP_NUM_THREADS=16 time ./md
OMP_NUM_THREADS=8  time ./md
OMP_NUM_THREADS=4  time ./md
OMP_NUM_THREADS=2  time ./md
OMP_NUM_THREADS=1  time ./md

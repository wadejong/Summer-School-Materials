#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=40 --cpus-per-task=1 --time=00:20:00 --job-name=mdmpi -p debug-40core

mpirun -np 32 ./md
mpirun -np 16 ./md
mpirun -np  8 ./md
mpirun -np  4 ./md
mpirun -np  2 ./md
mpirun -np  1 ./md



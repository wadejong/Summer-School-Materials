#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=40 --cpus-per-task=1 --time=00:20:00 --job-name=vmcmpi -p debug-40core

time mpirun  ./vmc

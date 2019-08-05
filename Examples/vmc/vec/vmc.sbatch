#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --time=00:20:00 --job-name=vmcvec -p debug-40core

time ./vmc

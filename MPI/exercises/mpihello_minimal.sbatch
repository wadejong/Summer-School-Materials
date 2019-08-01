#!/bin/bash
#SBATCH --nodes=2 --ntasks-pepr-node=40 --cpus-per-task=1 --time=00:05:00 --job-name=test -p debug-40core

mpirun ./mpihello



#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=40 --time=00:20:00 --job-name=scfomp -p debug-40core

OMP_NUM_THREADS=32 time ./scf ala2.xyz
OMP_NUM_THREADS=16 time ./scf ala2.xyz
OMP_NUM_THREADS=8  time ./scf ala2.xyz
OMP_NUM_THREADS=4  time ./scf ala2.xyz
OMP_NUM_THREADS=2  time ./scf ala2.xyz
OMP_NUM_THREADS=1  time ./scf ala2.xyz

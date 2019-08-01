#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=40 --cpus-per-task=1 --time=00:20:00 --job-name=scfmpi -p debug-40core

time mpirun -np 32 ./scf ala2.xyz
time mpirun -np 16 ./scf ala2.xyz
time mpirun -np  8 ./scf ala2.xyz
time mpirun -np  4 ./scf ala2.xyz
time mpirun -np  2 ./scf ala2.xyz
time mpirun -np  1 ./scf ala2.xyz

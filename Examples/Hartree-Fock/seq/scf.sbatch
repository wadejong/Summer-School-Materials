#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --time=00:20:00 --job-name=scfseq -p debug-40core

time ./scf ala2.xyz

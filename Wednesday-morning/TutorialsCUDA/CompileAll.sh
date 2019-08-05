#!/bin/bash

for PROG in a b c d e ; do
  nvcc -o program_i${PROG}  program_i${PROG}.cu
  nvcc -o program_ii${PROG} program_ii${PROG}.cu
done

for PROG in iif iiia iiia_solved iva iva_solved ; do
  nvcc -o program_${PROG}  program_${PROG}.cu
done

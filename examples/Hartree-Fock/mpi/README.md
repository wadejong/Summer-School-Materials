Incremental transformation to derive the MPI version.

We will parallelize in exactly the same way as the OpenMP version


1. Introduce MPIInit and Finalize get rid of excess print and guard all remaining print statements with if(rank==0)

    * Oh, and insert some total timing again

2. Broadcast command line argument to all processes 

    * What size buffer should we use for broadcasting?

    * [here assume that everyone will have access to the file using the same name ... not usually a valid assumption]

3. Use the same count/mod mechanism and a global sum

    * Ugh -- how to do the global sum using an eigen matrix?

    * Don't forget to do all reduce.

    * Paranoia suggests also broadcasting the orbitals in case of random phases

Exercises:

* repeat the parallelization starting from the sequential version

* change it so only process 0 reads from the file and broadcasts data to everyone

* parallelize the 1-electron pieces

* develop a performance model for the program

* introduce Schwarz screening


Hexane: times from old login node
 | build time (s) | tital time (s) | nproc |
 |: -------------:|:--------------:|:-----:|
 |1.51            |1.80            | 64    |
 |2.66            |2.94            | 32    |
 |4.92            |5.20            | 16    |
 |6.22            |6.48            | 12    |
 |8.88            |9.10            | 8     |
 |16.98           |17.19           | 4     |
 |33.48           |33.68           | 2     |
 |66.70           |66.90           | 1     |



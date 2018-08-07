Incremental transformation from sequential to parallel

1. Insert MPI_Init and finalize, restrict printing to node zero only --- but still running sequentially

2. Mmm --- now need to think.  How are we going to parallelize this?

    * Let's replicate the particles since distributing them is much more complicated (and more efficient!).

    * Since evaluation of the forces is driven by the neighborlist we can have each node compute just a subset of the neighborlist and then do a global sum so that everyone has all of the forces.

    * What are the potential problems here?

    * Prolly need to broadcast the initial coords and velocities to ensure everyone starting from the same place
   
3. Restrict computation of the neighbor list to a subset of particles per node and introduce a global sum into the forces

    * Oh, and don't forget to global sum the virial and pe also

These times with all processes within a node

|force   |neigh    |total  | nproc |
|:------:|:-------:|:-----:|:-----:|
|1.91s  |21.09s  |44.37s  |1|
|1.18s  |10.52s  |22.43s  |2| 
|5.83s  | 5.38s  |11.67s  |4| 
|3.15s  | 2.77s  | 6.22s  |8| 
|2.44s  | 1.92s  | 4.67s  |12|
|3.02s  | 1.48s  | 4.75s  |16|

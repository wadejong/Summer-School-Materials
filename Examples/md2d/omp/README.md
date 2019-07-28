Incremental construction of the OpenMP version

How to parallelize this?  We could do it the same way as the MPI version but it would be nice to explore other alternatives.

Also, if we do it differently to MPI (explicitly if we can exploit finer grain parallelism) then perhaps we can make more easily make an efficient hybrid OpenMP+MPI program.

In the MPI program we parallelized the outer loop of the neighborlist generation, so in this code perhaps we could parallelize both loops. But what does this mean about threads adding into a shared force vector?  What are the implications about increased memory traffic as multiple threads read the data? What about caches shared by multiple threads or cache coherency traffic between different caches? 

Sigh --- ideally we would have a performance model.

Note that STL data structures are NOT thread safe (operations that read these structures can be used by threads BUT modifying the data structure safely in a threaded program is actually much harder than you might think).

Note that the neighborlist is assembled in a STL list that is dynamically grown.  This requires lots of small memory allocations and growing the list is not thread safe and forces traversal to be sequential.  We could switch to a vector?

[Aside: a) In C++ STL what is the difference between a list and a vector?  b) How is a vector actually extended when there is not enuf space for another element to be pushed onto the back?  c) How is an STL list indexed/traversed?]


1. Let's have a neighbor list per thread.  Hopefully this improves cache locality, and using a vector should reduce memory allocation and traversal overheads.

    * First parallelize the neightbor list but before returning assemble a full list so the rest of the code functions unchanged.

2. Now return the parallel neighbor lists ... must be careful to not have a memory leak or do excess copying of data.

    * Use critical sections to protect update of f

3. Sigh.  Why are the forces so slooooow even with 1 thread?

    * Overhead of the critical section.  Switch to using replicated arrays.  Why is this eventually not scalable?

4.  Sigh.  Why does the neighbor list not scale well?

    * We are doing n**2 if tests per thread.  This is expensive.

    * Parallelize the outer loop then load balance after the lists are made


These times while running on the login node -- you can do much better on a dedicated node

|force   |neigh    |total  | nproc |
|:------:|:-------:|:-----:|:-----:|
|20.68s  |20.32s  |41.14s  |1 |
|10.46s  |10.88s  |21.54s  |2 |
| 5.44s  | 5.62s  |11.31s  |4 |
| 3.79s  | 3.84s  | 7.86s  |6 |
| 2.98s  | 3.06s  | 6.27s  |8 |
| 2.57s  | 2.69s  | 5.50s  |10|
| 2.28s  | 2.25s  | 4.76s  |12|
| 2.21s  | 2.19s  | 4.65s  |14|
| 2.98s  | 2.02s  | 5.26s  |16|



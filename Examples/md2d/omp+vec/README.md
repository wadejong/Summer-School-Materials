# OpenMP + Vectorized version

Started from the OpenMP version and did the following
* stored the coordinates, forces, and velocities in contiguous arrays (this is a common issue --- changing from a vector of structs into a struct of vectors)
* changed the neighborlist from a list of interacting pairs into for given `i` a list of interacting `j` so that we have a vector of independent `j` to vectorize over
* in both forces and neighbor list computation split the loop into vectorizable sections over contiguous arrays with gather/scatter operations before/after --- this step finally sped things up (about 2x for forces and 5x for neighborlist)
* rearranged the forces computation to reduce the number of reciprocals (since that vectorizes but not super efficiently) --- about a 20% reduction of cost in the forces
* introduced [Morton ordering](https://en.wikipedia.org/wiki/Z-order_curve) to increase locality --- a modest speedup of the forces but more helpful for the neighborlist
* parameterized the type as `FLOAT` so can play with `single` and `double`

For better timing increase quadruple the number of atoms (`natom`) and double the side (`L`) of the cell. Also, on `sn-mem` best timing obtained with something of the form
~~~
OMP_PLACES="cores" OMP_PROC_BIND="close" OMP_NUM_THREADS=18 ./md
~~~


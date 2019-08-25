# Solution of Laplace's equation on a 2D mesh --- MPI+OpenMP+SIMD

This program solves Laplace's equation in 2D using a 5-point stencial
and the now old-fashioned and inefficient [Gauss-Seidel](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) or
successive over relaxation ([SSOR](https://en.wikipedia.org/wiki/Successive_over-relaxation)) method --- much
more efficient modern solvers exist.  However, we use this iteration
since the code is simple and it nicely illustrates use of the point-to-point
communication routines in MPI.

On the square domain <a href="https://www.codecogs.com/eqnedit.php?latex=\Omega=[-\pi,\pi]^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Omega=[-\pi,\pi]^2" title="\Omega=[-\pi,\pi]^2" /></a>
we wish to solve Laplace's equation 

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla^2&space;u(x,y)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nabla^2&space;u(x,y)&space;=&space;0" title="\nabla^2 u(x,y) = 0" /></a>

subject to the boundary conditions 

<a href="https://www.codecogs.com/eqnedit.php?latex=u(x,y)=f(x,y)\&space;\mbox{for}\&space;(x,y)\in\partial\Omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u(x,y)=f(x,y)\&space;\mbox{for}\&space;(x,y)\in\partial\Omega" title="u(x,y)=f(x,y)\ \mbox{for}\ (x,y)\in\partial\Omega" /></a> (i.e., on the edges of the square). 

We approximate the Laplacian using a standard stencil 

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla^2&space;u(x,y)&space;\approx&space;\left(&space;u(x-h,y)&space;&plus;&space;u(x&plus;h,y)&space;&plus;&space;u(x,y-h)&space;&plus;&space;u(x,y&plus;h)&space;-&space;4&space;u(x,y)&space;\right)&space;/&space;h^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nabla^2&space;u(x,y)&space;\approx&space;\left(&space;u(x-h,y)&space;&plus;&space;u(x&plus;h,y)&space;&plus;&space;u(x,y-h)&space;&plus;&space;u(x,y&plus;h)&space;-&space;4&space;u(x,y)&space;\right)&space;/&space;h^2" title="\nabla^2 u(x,y) \approx \left( u(x-h,y) + u(x+h,y) + u(x,y-h) + u(x,y+h) - 4 u(x,y) \right) / h^2" /></a>.  

Setting this to zero, discretizing (evaluating on a grid) the solution so that <a href="https://www.codecogs.com/eqnedit.php?latex=u_{ij}&space;=&space;u(i&space;h-\pi,&space;j&space;h-\pi)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{ij}&space;=&space;u(i&space;h-\pi,&space;j&space;h-\pi)" title="u_{ij} = u(i h-\pi, j h-\pi)" /></a> (with <a href="https://www.codecogs.com/eqnedit.php?latex=h=2\pi/(N-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h=2\pi/(N-1)" title="h=2\pi/(N-1)" /></a> for *N* grid points including those on the boundary), and interpreting the equation as a fixed point iteration yields (with *n* as the iteration index)

<a href="https://www.codecogs.com/eqnedit.php?latex=u^{n&plus;1}_{i,j}&space;=&space;\left(u^n_{i-1,j}&space;&plus;&space;u^n_{i&plus;1,j}&space;&plus;&space;u^n_{i,j-1}&space;&plus;&space;u^n_{i,j&plus;1}&space;\right)/4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u^{n&plus;1}_{i,j}&space;=&space;\left(u^n_{i-1,j}&space;&plus;&space;u^n_{i&plus;1,j}&space;&plus;&space;u^n_{i,j-1}&space;&plus;&space;u^n_{i,j&plus;1}&space;\right)/4" title="u^{n+1}_{i,j} = \left(u^n_{i-1,j} + u^n_{i+1,j} + u^n_{i,j-1} + u^n_{i,j+1} \right)/4" /></a>

Overrelaxation tries to accelerate convergence by taking a bigger step.

<a href="https://www.codecogs.com/eqnedit.php?latex=u^{n&plus;1}_{i,j}&space;\leftarrow&space;u^{n&plus;1}_{i,j}&space;&plus;&space;\omega&space;\left(u^{n&plus;1}_{i,j}&space;-&space;u^{n}_{i,j}&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u^{n&plus;1}_{i,j}&space;\leftarrow&space;u^{n&plus;1}_{i,j}&space;&plus;&space;\omega&space;\left(u^{n&plus;1}_{i,j}&space;-&space;u^{n}_{i,j}&space;\right)" title="u^{n+1}_{i,j} \leftarrow u^{n+1}_{i,j} + \omega \left(u^{n+1}_{i,j} - u^{n}_{i,j} \right)" /></a>

The program chooses the optimum value for <a href="https://www.codecogs.com/eqnedit.php?latex=\omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega" title="\omega" /></a> for each grid spacing.

This iteration rapidly damps high-frequency errors but is very slow at eliminating low-frequency errors.  Since low-frequencies on a fine mesh appear as high-frequency on a coarse mesh, modern solvers use multigrid techniques (and more powerful iterative techniques) to cycle between fine and coarse meshes to greatly accelerate convergence.  We stop short of the complexity of a full multigrid solver and only solve on a sequence of increasingly fine meshes.

![grid](mesh.gif  "Grid")

We are solving for the interior `(N-2)*(N-2)` points in the mesh.  Starting with a coarse mesh, we initialize the interior points to zero and set the boundary values.  We then iterate until the change between iterations is acceptably small.  Next, we interpolate onto a finer mesh with double the number of grid points, set the boundary values again, and then solve on that mesh.   This is by default done for four meshes.

The available parallelism is shown by coloring of the mesh points as red/black --- you can see that updating any red point only requires values from the neighboring black points, and vice versa.  Thus, we can update all of the red points in parallel, and then update all of the black points again in parallel.

To parallelize the calculation using MPI the proceessors are arranged into a 2D grid (`nrow_P,ncol_P`) that is depicted in gray in the figure. The rows and columns of the actual mesh are partitioned accordingly with the patch owned by each process identified by coordinate of the top-left cell in its local patch.

Since updating a value on the edge of the grid local to a processor requires data from the neighboring processor, these rows and edges must be exchanged before each update.  This is done in the routine `Exchange` --- this routine is actually sending twice as much data as necessary (it sends both red and black points each time but only needs to send one set at a time), and could also be made a bit more efficient by using asynchronous communication to decouple the order of execution each process.

The `i` loop is parallelized using OpenMP.

The `j` loop is vectorized.  This inner loop has stride 2 --- we can anticipate a factor of two speed up if we made it unit stride by storing the red and black points in separate arrays.  However, it would make the example code even more complicated.

Note that MPI is initialized to let it know we might be using threads but that only the main thread is going to be calling MPI routines while the other threads are sleeping.

With `NGRID=3844` these are timings from the four socket `sn-mem` SkyLake node (with the evironment variable `I_MPI_PIN_DOMAIN=socket` to bind all threads from an MPI process to the same socket).  The time is the operation time that includes the time taken for exchanging data (i.e., the communication cost).

P=#MPI processes

T=#OMP threads per process

Speedup = time / time for 1 process with 1 thread

Efficiency = Speedup/(P*T)

|  P  |  T  | Time(s) | Speedup | Efficiency |
|-----|-----|---------|---------|------------|
|1|1|115.40|1.0|100.0%|
|1|2|59.40|1.9|97.1%|
|1|3|35.50|3.3|108.4%|
|1|4|29.80|3.9|96.8%|
|1|6|22.90|5.0|84.0%|
|1|8|20.30|5.7|71.1%|
|1|10|18.10|6.4|63.8%|
|1|12|16.90|6.8|56.9%|
|1|14|16.30|7.1|50.6%|
|1|16|16.00|7.2|45.1%|
|1|18|15.70|7.4|40.8%|
|2|1|52.30|2.2|110.3%|
|2|2|25.40|4.5|113.6%|
|2|4|13.40|8.6|107.6%|
|2|8|8.42|13.7|85.7%|
|2|12|6.86|16.8|70.1%|
|2|16|6.22|18.6|58.0%|
|4|1|21.60|5.3|133.6%|
|4|2|11.50|10.0|125.4%|
|4|4|5.94|19.4|121.4%|
|4|8|3.43|33.6|105.1%|
|4|12|2.13|54.2|112.9%|
|4|15|1.70|67.9|113.1%|
|4|16|1.72|67.1|104.8%|
|8|8|1.67|69.1|108.0%|
|12|5|1.57|73.5|122.5%|
|15|4|1.71|67.5|112.5%|
|30|2|1.61|71.7|119.5%|
|40|1|2.42|47.7|119.2%|
|60|1|1.49|77.4|129.1%|


The following times are without binding threads to a socket.  This enables a single process OpenMP to access the full memory bandwidth of the system. The largest mesh at 118 MBytes is much larger than the L2 or L3 caches, and hence this application is limited by memory bandwidth.  Comparing the below with the above times, you can see that about 4 threads start to saturate the single-socket memory bandwidth. Also used `OMP_PROC_BIND=spread`.

|  P  |  T  | Time(s) | Speedup | Efficiency |
|-----|-----|---------|---------|------------|
|1|1|115.40|1.0|100.0%|
|1|2|59.80|1.9|96.5%|
|1|3|33.50|3.4|114.8%|
|1|4|29.60|3.9|97.5%|
|1|5|22.60|5.1|102.1%|
|1|6|16.00|7.2|120.2%|
|1|8|11.30|10.2|127.7%|
|1|12|7.91|14.6|121.6%|
|1|15|6.94|16.6|110.9%|
|1|20|5.10|22.6|113.1%|
|1|24|4.80|24.0|100.2%|
|1|30|4.24|27.2|90.7%|
|1|40|3.33|34.7|86.6%|
|1|60|2.83|40.8|68.0%|
|1|1|115.40|1.0|100.0%|
|2|1|52.70|2.2|109.5%|
|3|1|29.20|4.0|131.7%|
|4|1|27.30|4.2|105.7%|
|5|1|20.80|5.5|111.0%|
|6|1|16.90|6.8|113.8%|
|8|1|12.30|9.4|117.3%|
|12|1|8.06|14.3|119.3%|
|15|1|6.10|18.9|126.1%|
|20|1|5.06|22.8|114.0%|
|24|1|4.38|26.3|109.8%|
|30|1|4.00|28.9|96.2%|
|40|1|3.19|36.2|90.4%|
|60|1|1.98|58.3|97.1%|















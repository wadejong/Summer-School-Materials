---
title: "Intro to Hybrid Parallelization"
---

## Creating a hybrid MPI-OpenMP parallelized code

We will add MPI parallelization to the OpenMP-parallelized code from a previous example.

**$ cd example1\
make\
qsub run.sh**

>    30.00     3973.40314   -19398.90851  -15425.50536    0.495   0.03703538  \
times:  force=3.92s  neigh=4.49s  total=8.53s

Add an include statement for `mpi.h`:

``` cpp
#include <mpi.h>
```

Also add lines to initialize and finalize MPI:

``` cpp
int nproc, me;
bool PRINT = false;

...

int main(int argc, char** argv) {
    if (MPI_Init(&argc,&argv) != MPI_SUCCESS)
      throw "MPI init failed";

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    PRINT = (me == 0);

    ...

    MPI_Finalize();
    return 0;
}
```

Then, ensure that rank 0 does all of the printing:

``` cpp
        if ((step%50)==0 && PRINT)
            std::cout << "optim: " <<  pe << " " <<  dt << std::endl;
...
    if (PRINT) std::cout << "Time step " << dt << std::endl;
...
            if (step == nprint && PRINT) {
                printf("\n");
                printf("    time         ke            pe             e            T          P\n");
			    printf("  -------    -----------   ------------  ------------    ------    ------\n");
            }
...
            if (PRINT) printf("%9.2f   %12.5f   %12.5f  %12.5f %8.3f %12.8f %c\n",
                   step*dt, kinetic_energy, potential_energy, energy, temp,
                   pressure, scaling[vscale!=1.0]);
...
    if (PRINT) printf("times:  force=%.2fs  neigh=%.2fs  total=%.2fs\n",
           time_force, time_neigh, time_total);
```

In order to ensure that all of the ranks have the same initial coordinates, we will broadcast the coordinates and velocities from rank 0.
Do this just before the line that reads `// Relax the initial random guess`

``` cpp
    // Broadcast the coordinates and velocities
    if (MPI_Bcast((void*) &coords[0], natom*sizeof(xyT), MPI_BYTE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) 
      throw("broadcast of coords failed");
    if (MPI_Bcast((void*) &v[0], natom*sizeof(xyT), MPI_BYTE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) 
      throw("broadcast of vels failed");
```

Now modify the neighborlist generation so that each rank has only a subset of the processes.

``` cpp
void neighbor_list(const coordT& coords, thrneighT& thrneigh) {
    ...
#pragma omp parallel default(none) shared(thrneigh, coords, nproc, me)
    {
        ...
        for (int i=(me*nthread)+ithr; i<natom; i+=nproc*nthread) {
...
coordT forces(const thrneighT& thrneigh, const coordT& coords, double& virial, double& pe) {
...
    // Reduce over MPI ranks                                                                                                                  
    coordT ftotal(natom, xyT(0.0,0.0));
    MPI_Allreduce(&f[0], &ftotal[0], 2*natom, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double virial_total, pe_total;
    MPI_Allreduce(&virial, &virial_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pe, &pe_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    virial = virial_total; // To return values via the argument list                                                                          
    pe = pe_total;

    time_force += omp_get_wtime() - start;
    return ftotal;
```

>    30.00     3973.40313   -19398.90850  -15425.50537    0.495   0.03703538  \
times:  force=1.07s  neigh=0.82s  total=2.01s



# Performance Analysis

Now that we have a hybrid MPI-OpenMP code, let's test how the performance of the code changes as we change the degree to which we rely on MPI or OpenMP.
Here are the timings I got when running on 24 cores:

|               | Forces | Neighbors | Total |
|:-------------:|--------|-----------|-------|
| 1 MPI, 24 OMP |   1.55 |      1.00 |  2.74 |
| 2 MPI, 12 OMP |   1.24 |      0.84 |  2.22 |
| 3 MPI, 8 OMP  |   1.47 |      0.82 |  2.41 |
| 4 MPI, 6 OMP  |   1.07 |      0.82 |  2.01 |
| 6 MPI, 4 OMP  |   1.04 |      0.82 |  1.99 |
| 8 MPI, 3 OMP  |   1.10 |      0.79 |  2.02 |
| 12 MPI, 2 OMP |   1.04 |      0.79 |  1.97 |
| 24 MPI, 1 OMP |   1.04 |      0.75 |  1.91 |



# Memory analysis

So if running with pure MPI gives us the best results, why would we ever run with OpenMP?
Keep in mind that time is only one of the factors that determines what calculations we are able to run - another import factor is memory availability.

We can measure the amount of memory used by a process with the `/usr/bin/time` command.
When this command is issued followed by another command, it will report resource usage statistics on the second command.
For example:

**$ /usr/bin/time -v ls**

>    Command being timed: "ls"\
    User time (seconds): 0.00\
    System time (seconds): 0.00\
    Percent of CPU this job got: 100%\
    Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.00\
    Average shared text size (kbytes): 0\
    Average unshared data size (kbytes): 0\
    Average stack size (kbytes): 0\
    Average total size (kbytes): 0\
    Maximum resident set size (kbytes): 1040\
    Average resident set size (kbytes): 0\
    Major (requiring I/O) page faults: 0\
    Minor (reclaiming a frame) page faults: 330\
    Voluntary context switches: 0\
    Involuntary context switches: 26\
    Swaps: 0\
    File system inputs: 0\
    File system outputs: 0\
    Socket messages sent: 0\
    Socket messages received: 0\
    Signals delivered: 0\
    Page size (bytes): 4096\
    Exit status: 0

"Maximum resident set size" tells us the maximum amount of memory used by the command.
Using this, we can determine how much memory our code is using.
There is unfortunately one complication that arises due to our use of MPI.
We only want to get statics on a single rank, so we will need a small script to ensure that this is what happens.
You will find this script in the `example1` directory, titled `exec_wrapper_mpi.sh`.
Its contents are:

```
#!/bin/bash
export MV2_ENABLE_AFFINITY=0
exec=$1
if [ $PMI_RANK -eq 0 ]; then
/usr/bin/time -v $exec
else
$exec
fi
```

We will want to add this script to `run.sh`:

```
#!/bin/bash
#PBS -l nodes=1:ppn=24,walltime=00:10:00
#PBS -q molssi
#PBS -N example1
#PBS -j oe

mpirun -n 24 ./exec_wrapper_mpi.sh ./md > output
```

Now run a series of calculations with different numbers of MPI tasks and OpenMP threads, just like you did in the last section.
This time, take note of the "Maximum resident set size" that is reported by the `/usr/bin/time` command.
Then, compute the total amount of memory required by the calculation, which is the "Maximum resident set size" times the number of MPI tasks.
Here are my results, with the sizes converted to MB:

|               | Rank Size | Total Size |
|---------------|-----------|------------|
| 1 MPI, 24 OMP |        32 |         32 |
| 2 MPI, 12 OMP |        21 |         43 |
| 3 MPI, 8 OMP  |        17 |         52 |
| 4 MPI, 6 OMP  |        18 |         72 |
| 6 MPI, 4 OMP  |        23 |        138 |
| 8 MPI, 3 OMP  |        21 |        167 |
| 12 MPI, 2 OMP |        17 |        209 |
| 24 MPI, 1 OMP |        16 |        386 |

Running in pure MPI mode is a little bit faster than running in pure OpenMP mode, but the MPI calculation takes more than 10x as much memory!
In many cases, it may be necessary to run with less MPI and more OpenMP, simple in order to fit the calculation in the available memory.



# OpenMP, MPI-Style


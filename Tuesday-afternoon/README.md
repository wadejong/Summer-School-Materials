
# Introduction to message passing with MPI

## 1. Outline

1.  Big picture
1.  Hello world in *just* 20 minutes
1.  Sending messages --- point to point communication
1.  Communicators 
1.  Reasoning about performance
1.  Communicators in more detail
  
## 2. Big picture


![distmem](hybrid_mem.gif  "Distributed memory")


## 3. Hello world 

#### Essential elements
1. Incremental transformation
2. Compiling and linking
3. Simple initialization
3. MPI interface convention and errors
4. Communicators
5. Process rank
6. Running
    * Interactively and in batch

### Writing hello world

Start from sequential version `hello.cc`
~~~
    #include <iostream>
    int main() {
        std::cout << "Hello" << std::endl;
        return 0;
    }
~~~
Build the sequential version with `make hello` or `icpc -o hello hello.cc`.

#### Required elements of all MPI programs

* Include `mpi.h` --- older versions of some MPI implementations required it be the first header
* Initialize MPI --- usually the first line of your main program will be similar to the following.
```
    if (MPI_Init(&argc,&argv) != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
```
* Finalize MPI --- usually the penultimate line of your main program will be
```
    MPI_Finalize();
```
* Initializing MPI gives us access to the default communicator (`MPI_COMM_WORLD`)
    * An intra communicator encapsulates all information needed for a group of processes to communicate with each other
    * We will look at communicators in more detail soon --- for now we just need to get the number of processes and the rank (`0,1,2,...`) of the current process.
* Note how MPI wants access to the command line arguments (so we must modify the signature of `main`). You can use `NULL` instead of `argv` and `argc` but passing arguments to MPI is very useful.

#### Error detection and exit
* MPI functions return `MPI_SUCCESS` on success or an error code (see documentation) on failure.
* To abort execution you cannot just `exit` or `return` because there's lots of clean up that needs to be done when running in parallel --- a poorly managed error can easily waste 1000s of hours of computer time.


The new version (`mpihello.cc`) looks like this
~~~
    #include <mpi.h>
    #include <iostream>
    
    int main(int argc, char** argv) {
        if (MPI_Init(&argc,&argv) != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
        
        int nproc, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        
        std::cout << "Hello from process " << rank << " of " << nproc << std::endl;
        
        MPI_Finalize();
        return 0;
    }
~~~

#### Compiling MPI programs

Build your parallel version with `make mpihello1` or `mpiicpc -o mpihello1 mpihello1.cc`

* MPI provides wrappers for compiling and linking programms because there's a lot of machine specific stuff that must be done.
* For the summer school we are using the Intel C++ compiler and MPI library so we use the command `mpiicpc` (`mpiifort` for the Intel fortran, `mpiicc` for Intel C) but more commonly you might be using the GNU stack (`mpicxx`, `mpicc`, `mpifort`, etc.)

#### Running MPI programs

You can run the program sequentially just like any other program.

To run it in parallel we must use the use the `mpirun` command (this is system dependent and a common variant is `mpiexec`) since again there's lots of machine dependent stuff that needs doing.  At its simplest we must tell MPI how many processes we want to use --- here we use 4.
```
    mpirun -np 4 ./mpihello1
```
Your output might look like
```
Hello from process 1 of 4
Hello from process 0 of 4
Hello from process 3 of 4
Hello from process 2 of 4
```
but more likely will look something like
```
Hello from process Hello from process 1 of 4
Hello from process 2 of 4
3 of 4
Hello from process 0 of 4
```

We used four processes on the local machine (e.g., your laptop or the cluster login node).  More typically, we want to use multiple computers and most clusters use a batch system to
* time share the computers in the cluster
* queue jobs according to priority, resource needs, etc.

**EXAMPLE BATCH JOB HERE**

You can also provide to `mpirun` a hostfile that tells it which computers to use --- on most clusters this is rarely neccessary.


#### Some mpirun options

* Separating and labelling output
* ???

#### 


##  Sending and receiving messages --- point to point communication

#### 





Sometimes it is confusing where stuff is actually running, in part because every job scheduler and resource manager seems to be set up differently, so we modify the program to also print out the name of the machine each process is running on



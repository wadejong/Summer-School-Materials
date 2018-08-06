
# Introduction to message passing with MPI

## 1. Outline

1.  Big picture
1.  Useful links
1.  Hello world in *just* 20 minutes
1.  Sending messages between processes
1.  Global operations
1.  Review of the 6-8 essential MPI operations
1.  Communicators and groups
1.  Reasoning about performance
1.  Debugging, etc.
1.  Additional material
1.  Exercises
  
## 2. Big picture

Motivations:
* Performance

* Price-performance

* Correctness

The only cost-effective path to massive performance is connecting together multiple commodity computers via a high-performance network.  In present technologies each computer will contain multiple CPUs (cores) that share memory. They might also contain GPUs or other accelerators.

![distmem](images/hybrid_mem.gif  "Distributed memory")

We wish to program this cluster of computers to collaborate in the solution of a single science problem.  The challenge is that the processes do not share any memory (the memory and the data it contains is distributed across the cluster), and potentially not even a file system.  This is a classic problem in concurrent systems, and the communicating sequential processes (CSP; see references below) model provides a rigorous solution with provable properties.

The essential idea is exactly how a team of humans distributed across the planet would solve a problem via email --- they would send messages to each other or the whole team until everyone had the data they needed to solve their part of the problem.  Partial or full results would be similarly communicated.  By formalizing this approach and introducing concepts such as ordering and message types, CSP enables certain styles of writing message programs to be proven to be correct and safe (such as in the sense of needing bounded buffers).

Moreover, message-passing programs are intrinsically safer and in some sense easier to write than shared-memory programs.  In a shared-memory program any process/thread can modify any shared data at any time.  It takes substantial programmer discipline and good design to avoid race conditions and myriad other bugs and performance problems.  In contrast, in a message-passing program no data is shared and so there are no such problems.

Finally, even within a shared-memory computer a message-passing program can sometimes run faster than a shared-memory program.  This is because it is expensive (in time and energy) to share data between processors and it is hard to deisgn a shared-memory program that minimizes the amount of data being touched by multiple processors.  In contrast, a message-passing program shares no data, and all memory references are local to that process.  Indeed, a high-performance shared-memory program can sometimes start to resemble a message-passing program.

The Message Passing Interface (MPI) is now the defacto standard for message passing in high-performance computing, with multiple implementations from the community and computer vendors.

The objective of this brief tutorial is to introduce key elements of MPI and its practical use within the molecular sciences.

References:
* CSP --- https://en.wikipedia.org/wiki/Communicating_sequential_processes

* CSP --- http://www.usingcsp.com/cspbook.pdf

* MPI standard --- https://www.mpi-forum.org/docs/


## 3. Useful links

* [MPI interface](https://www.mpich.org/static/docs/v3.2/)

* [MPI tutorial](https://computing.llnl.gov/tutorials/mpi/) from LLNL --- excellent, thorough with good links

* [MPI tutorial](https://htor.inf.ethz.ch/teaching/mpi_tutorials/ppopp13/2013-02-24-ppopp-mpi-basic.pdf) --- MPI for dummies

* [Intel MPI](https://software.intel.com/en-us/intel-mpi-library/documentation) documentation

* [Intel C++](https://software.intel.com/en-us/cpp-compiler-18.0-developer-guide-and-reference) compiler documentation


## 3. Hello world 

### Essential elements:
1. Incremental transformation
2. Compiling and linking
3. Simple initialization
3. MPI interface convention and errors
4. Communicators
5. Process rank
6. Running interactively and in batch

### Writing hello world

Start from sequential version `hello.cc`
```
    #include <iostream>
    int main() {
        std::cout << "Hello" << std::endl;
        return 0;
    }
```
Build the sequential version with `make hello` or `icpc -o hello hello.cc`.

### Required elements of all MPI programs

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
    * An intra communicator encapsulates all information and resources needed for a group of processes to communicate with each other
    * We will look at communicators in more detail soon --- for now we just need to get the number of processes and the rank (`0,1,2,...`) of the current process.
* Note how MPI wants access to the command line arguments (so we must modify the signature of `main`). You can use `NULL` instead of `argv` and `argc` but passing arguments to MPI is very useful.

### Error detection and exit
* MPI functions return `MPI_SUCCESS` on success or an error code (see documentation) on failure.
* To abort execution you cannot just `exit` or `return` because there's lots of clean up that needs to be done when running in parallel --- a poorly managed error can easily waste 1000s of hours of computer time.


The new version (`mpihello.cc`) looks like this
```
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
```

### Compiling MPI programs

Build your parallel version with `make mpihello1` or `mpiicpc -o mpihello1 mpihello1.cc`

* MPI provides wrappers for compiling and linking programs because there's a lot of machine specific stuff that must be done.
* For the summer school we are using the Intel C++ compiler and MPI library so we use the command `mpiicpc` (`mpiifort` for the Intel FORTRAN, `mpiicc` for Intel C) but more commonly you might be using the GNU stack (`mpicxx`, `mpicc`, `mpifort`, etc.)

### Running MPI programs

You can run the program sequentially just like any other program.

To run it in parallel we must use the use the `mpirun` command (this is system dependent and a common variant is `mpiexec`) since again there's lots of machine dependent stuff that needs doing.  At its simplest we must tell MPI how many processes we want to use --- here we use 4.
~~~
    mpirun -np 4 ./mpihello1
~~~

Your output might look like
~~~
    Hello from process 1 of 4
    Hello from process 0 of 4
    Hello from process 3 of 4
    Hello from process 2 of 4
~~~
but more likely will look something like
~~~
    Hello from process Hello from process 1 of 4
    Hello from process 2 of 4
    3 of 4
    Hello from process 0 of 4
~~~

We used four processes on the local machine (e.g., your laptop or the cluster login node).  More typically, we want to use multiple computers.  You can manually provide to `mpirun` a hostfile that tells it which computers to use --- on most clusters this is rarely necessary since a batch system is used to
* time share the computers in the cluster
* queue jobs according to priority, resource needs, etc.


Here's an example batch job (`mpihello.pbs`) for SeaWulf annotated so show what is going on:
~~~
    #!/bin/bash
    #PBS -l nodes=2:ppn=24,walltime=00:02:00
    #PBS -q molssi
    #PBS -N hello
    #PBS -j oe

    # Above says:
    # - job has 2 (dedicated) nodes with 24 processes per node with a 2 minute max runtime
    # - use the molssi queue
    # - name the job "hello"
    # - merge the standard output and error into one file

    # Output should appear in the file "<jobname>.o<jobnumber>" in the
    # directory from which you submitted the job

    # ================================================
    # If this is not in your .bashrc it needs to be here so that your job
    # uses the same compilers/libraries that you compiled with
    source /gpfs/projects/molssi/modules-intel

    # This will change to the directory from which you submitted the job
    # which we assume below is the one holding the executable
    cd $PBS_O_WORKDIR

    # Uncomment this if you want to see other PBS environment variables
    # env | grep PBS

    # Finally, run the executable using $PBS_NUM_NODES*$PBS_NUM_PPN
    # processes spread across all the nodes
    mpirun ./mpihello

    # You can run more things below
~~~
But I find the comments distracting, so here is a minimal version.
~~~
     #!/bin/bash
     #PBS -l nodes=2:ppn=24,walltime=00:02:00
     #PBS -q molssi -N hello -j oe

     source /gpfs/projects/molssi/modules-intel
     cd $PBS_O_WORKDIR
     mpirun ./mpihello
~~~
You can copy and edit the file for your other jobs.  Note that other other systems running PBS will differ.

Submit the job from the directory holding your executable (or modify the batch script to use the full path to your executable)
~~~
    qsub mpihello.pbs
~~~

Useful PBS/Torque commands are
* `qstat` --- see all queued/running jobs
* `qstat -u <username>` --- to see just your jobs
* `qstat -f <jobid?>` --- to see detailed info about a job
* `qstat -Q` and `qstat -q` --- to see info about batch queues (for the summer school only `molssi` is available)
* `qdel <jobid>` --- to cancel a job


##  4. Sending and receiving messages --- point to point communication

### Essential elements
1. Process rank, message tag, MPI data type, communicator size
2. Blocking communication
3. One minimal set of six operations
3. Buffering and safe communication
4. Non-blocking communication
5. Other communication modes (synchronous send, buffered send)

A process is identified by its rank --- an integer `0,1,..,P-1` where `P` is the number of processes in the communicator (`P` is the size of the communicator).

Every message has a `tag` (integer) that you can use to uniquely identify the message.  This implements the CSP concept of message type --- MPI uses the term `tag` in order to distinguish from the datatype being sent (byte, integer, double, etc.).

In a conversation between a pair of processes, messages of the same `tag` are received in the order sent.

But messages from multiple processes can be interleaved with each other.


### Blocking communication

* When a blocking send function completes the buffer can immediately be reused without affecting the sent message.  Note that the receiving process may not necessarily have yet received the message.
* When a blocking recv function completes the received message is fully available in the buffer.

```
    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,  MPI_Comm comm)
```
* `buf` --- pointer to the start of the buffer being sent
* `count` --- number of elements to send
* `datatype` --- MPI data type of each element
* `dest` --- rank of destination process
* `tag`  --- message tag
* `comm` --- the communicator to use

```
    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
```
* `buf` --- pointer to the start of the buffer to receive the message
* `count` --- maximum number of elements the buffer can hold
* `datatype` --- MPI data type of each element
* `source` --- rank of source process ---  `MPI_ANY_SOURCE` matches any process
* `tag`  --- message tag (integer `>= 0`) --- `MPI_ANY_TAG` matches any tag
* `comm` --- the communicator to use
* `status` --- pointer to the structure in which to store status

The actual source, count and tag of the received message can be accessed from the status.
```
    status.MPI_TAG;
    status.MPI_SOURCE;
    MPI_Get_count( &status, datatype, &count );
```

There's data types for everything, and you can define you own including non-contiguous data structures:
|**MPI data type**  |**C data type**     |
|:------------------|:-------------------|
|MPI_BYTE           |8 binary digits     |
|MPI_CHAR           |char                |
|MPI_UNSIGNED_CHAR  |unsigned char       |
|MPI_SHORT          |signed short int	 |	 
|MPI_UNSIGNED_SHORT |unsigned short int	 |	 
|MPI_INT            |signed int          |
|MPI_UNSIGNED       |unsigned int	 |	 
|MPI_LONG           |signed long int	 |	 
|MPI_UNSIGNED_LONG  |unsigned long int	 |	 
|MPI_FLOAT          |float               |
|MPI_DOUBLE         |double              |
|etc.               |                    |
|MPI_PACKED	    |define your own with|
|                   |MPI_Pack/MPI_Unpack |


#### Exercise:

Write a program to send an integer (`=99`) from process 0 to process 1 and verify the value is correct.  If it is correct, then print "OK" and terminate correctly, otherwise abort.  Run your program.

Hint: start by copying `mpihello.cc`.

#### Exercise:

Write a program that has two processes exchanging a buffer of length `N` bytes.  Initialize the buffers to the process rank and verify the exchange happened correctly (i.e., the elements in the buffer received by process 1 should have the value 0).  Try `N=10**n` for `n=0,1,2,3,4,5,6,7,8` (i.e., go from small to very large messages).

There are several ways (both correct and incorrect) of writing this program.  You might try the simplest option first --- each process first sends its buffer and then receives its buffer.

Note that this is such a common operation that there is a special `MPI_Sendrecv` operation to make this less error prone, less verbose, and to enable optimizations.

#### Exercise:

Write a program to send an integer (`=99`) around a ring of processes (i.e., `0` sends to `1`, `1` sends to `2`, `2` sends to `3`, ..., `P-1` sends to `0`, with process 0 verifying the value is correct.  Your program should work for any number of processes greater than one.


### One minimal set of six operations

~~~
    MPI_Init
    MPI_Finalize
    MPI_Comm_size
    MPI_Comm_rank
    MPI_Send
    MPI_Recv
~~~

### Non-blocking (asynchronous) communication

* When a non-blocking send function completes, the user must not modify the send buffer until the request is known to have completed (e.g., using `MPI_Test` or `MPI_Wait`).

* When a non-blocking recv function completes, any message data is not completely available in the buffer until the request is known to have completed.

```
    int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,  MPI_Comm comm, MPI_Request *request)
    int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,  int tag, MPI_Comm comm, MPI_Request *request)
```
* `request` --- a pointer to structure that will hold the information and status for the request

```
    int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
    int MPI_Wait(MPI_Request *request, MPI_Status *status)
    int MPI_Cancel(MPI_Request *request)
```
* `request` --- a pointer to the request being tested/waited-upon/cancelled
* `flag` --- a pointer to an int that will be non-zero (true) if the operation has completed
* `status` --- a pointer to the structure in which status will be stored if the operation has completed
* See also `MPI_Waitall`, `MPI_Waitany` and `MPI_Waitsome` for waiting on multiple requests


### Other P2P communication modes

*  Buffered send --- provide sender-side buffering to ensure a send always completes and to make memory-management more explicit
*  Synchronous send --- completes on the sender-side when the receive has also completed
*  Ready send --- if you know a matching receive has already been posted this enables optimizations (and this style of programming is explicitly safe from memory/buffer issues)


### One-sided operations

**to be added**


## 5. Global operations

Many chemistry, materials, and biophysics applications are written without using any point-to-point communication routines.

### Essential elements
1. Broadcast
2. Reduction
3. Barrier
3. Gather and scatter
4. Non-blocking globals
5. Other communication modes (synchronous send, buffered send)

**to be completed**

### Another minimal set of six operations

~~~
    MPI_Init
    MPI_Finalize
    MPI_Comm_size
    MPI_Comm_rank
    MPI_Bcast
    MPI_Reduce
~~~
Or eight if you include
~~~
    MPI_Send
    MPI_Recv
~~~



##  6. Reasoning about performance

Essential concepts to reason about the performance of your message passing application

* Amdahl's law

* Speed up and efficiency

* Load balance

* Data balance

* Data distribution

* Latency and bandwidth of communication


### Amdahl's law

[Ahdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law) is both simple and brutal.  Dividing the execution time into a sequential component (*Ts*) and perfectly parallel program (*Tp*) the execution time on *P* processes is then

<img src="https://latex.codecogs.com/svg.latex?\Large&space;T(P)=Ts + \frac{Tp}{P}" title="Amdahl" />



<img src="https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />





##  Debugging, etc.


**to be added**

##  Additional concepts and material

* Inter communicators
* Toplogies
* 

**to be added**

## Exercises

**to be added**


# Introduction to message passing with MPI

1.  Big picture
1.  Useful links
1.  Hello world in *just* 20 minutes
1.  Sending messages between processes
1.  Global or collective operations
1.  Review of the 6-8 essential MPI operations
1.  Communicators and groups
1.  Reasoning about performance
1   Work distribution strategies
1.  Debugging, etc.
1.  Additional material
1.  Exercises
  
## 1. Big picture

Motivations:
* Performance

* Price-performance

* Correctness

The only cost-effective path to massive performance is connecting together multiple commodity computers via a high-performance network.  As we add more computers, we add more compute power, more memory, more memory bandwidth, more disk space, etc. In present technologies each computer will contain multiple CPUs (cores) that share memory. They might also contain GPUs or other accelerators.

![distmem](images/hybrid_mem.gif  "Distributed memory")

We wish to program this cluster of computers to collaborate in the solution of a single science problem.  The challenge is that the processes do not share any memory (the memory and the data it contains is distributed across the cluster), and potentially not even a file system.  This is a classic problem in concurrent systems, and the communicating sequential processes (CSP; see references below) model provides a rigorous solution with provable properties.  Sometimes performance is not our only concern --- for instance, you might have a problems with a massive amount of data that could be enabled by the aggregate memory of a cluster.

The essential idea is exactly how a team of humans distributed across the planet would solve a problem via email --- they would send messages to each other or the whole team until everyone had the data they needed to solve their part of the problem.  Partial or full results would be similarly communicated.  By formalizing this approach and introducing concepts such as ordering and message types, CSP enables certain styles of writing message programs to be proven to be correct and safe (such as in the sense of needing bounded buffers).

Moreover, message-passing programs are intrinsically safer and in some sense easier to write than shared-memory programs.  In a shared-memory program any process/thread can modify any shared data at any time.  It takes substantial programmer discipline and good design to avoid race conditions and myriad other bugs and performance problems.  In contrast, in a message-passing program no data is shared and so there are no such problems.

Finally, even within a shared-memory computer a message-passing program can sometimes run faster than a shared-memory program.  This is because it is expensive (in time and energy) to share data between processors and it is hard to deisgn a shared-memory program that minimizes the amount of data being touched by multiple processors.  In contrast, a message-passing program shares no data, and all memory references are local to that process.  Indeed, a high-performance shared-memory program can sometimes start to resemble a message-passing program.

The [Message Passing Interface (MPI)](https://www.mpi-forum.org/docs/) is now the defacto standard for message passing in high-performance computing, with multiple implementations from the community and computer vendors.

The objective of this brief tutorial is to introduce key elements of MPI and its practical use within the molecular sciences.

References:
* [CSP Wikipedia](https://en.wikipedia.org/wiki/Communicating_sequential_processes)

* [CSP Hoare book](http://www.usingcsp.com/cspbook.pdf)

* [MPI standard](https://www.mpi-forum.org/docs/)

* [Seawulf HPC FAQ etc.](https://it.stonybrook.edu/services/high-performance-computing)

* [Seawulf getting started guide](https://it.stonybrook.edu/help/kb/getting-started-guide)

* [Seawulf SLURM FAQ](https://it.stonybrook.edu/help/kb/using-the-slurm-workload-manager)

* [SLURM reference](https://slurm.schedmd.com/documentation.html)


## 2. Useful links

* [MPI interface](https://www.mpich.org/static/docs/v3.2/) --- from the MPICH team

* [MPI tutorial](https://computing.llnl.gov/tutorials/mpi/) from LLNL --- excellent, thorough, with good links

* [MPI tutorial(http://www.archer.ac.uk/training/course-material/2018/07/mpi-epcc/notes/MPP-notes.pdf) --- 2 days

* [MPI tutorial](https://htor.inf.ethz.ch/teaching/mpi_tutorials/ppopp13/2013-02-24-ppopp-mpi-basic.pdf) --- MPI for dummies

* [MPI lectures](http://wgropp.cs.illinois.edu/courses/cs598-s16) from Gropp at UIUC --- excellent and very detailed

* [Intel MPI](https://software.intel.com/en-us/intel-mpi-library/documentation) documentation

* [Intel C++](https://software.intel.com/en-us/cpp-compiler-18.0-developer-guide-and-reference) compiler documentation

* [GNU C/C++](https://gcc.gnu.org/onlinedocs/gcc-9.1.0/gcc) compiler documentation

* [MPI books](http://wgropp.cs.illinois.edu/usingmpiweb/) by Gropp good for intro and reference


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

Start from sequential version [`exercises/hello.cc`](https://github.com/wadejong/Summer-School-Materials/blob/master/MPI/exercises/hello.cc)
```c++
    #include <iostream>
    int main() {
        std::cout << "Hello" << std::endl;
        return 0;
    }
```
Build the sequential version with `make hello` or `icpc -o hello hello.cc` (Intel) or `g++ -o hello hello.cc` (GNU).

### Required elements of all MPI programs

* Include `mpi.h` --- older versions of some MPI implementations required it be the first header
* Initialize MPI --- by calling [`MPI_Init`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Init.html), or [`MPI_Init_thread`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Init_thread.html). Usually the first line of your main program will be similar to the following if you are not handling errors yourself (see just below)
```c++
    MPI_Init(&argc,&argv);
```
or this if you are handling errors
```c++
    if (MPI_Init(&argc,&argv) != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
```

* Finalize MPI --- by calling [`MPI_Finalize`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Finalize.html) usually the penultimate line of your main program will be
```c++
    MPI_Finalize();
```
* Initializing MPI gives us access to the default communicator (`MPI_COMM_WORLD`)
    * An intra communicator encapsulates all information and resources needed for a group of processes to communicate with each other.  For our simple applications we will always being `MPI_COMM_WORLD` but for real applications you should be passing a communicator into all of your routines to enable reuse and interoperability.
    * We will look at communicators in more detail soon --- for now we just need to get the number of processes (by calling [`MPI_Comm_size`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Comm_size.html)) and the rank (`0,1,2,...`) of the current process (by calling [`MPI_Comm_rank`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Comm_rank.html)).
* Note how MPI wants access to the command line arguments (so we must modify the signature of `main`). You can use `NULL` instead of `argv` and `argc` but passing arguments to MPI is very useful.

### Error detection and exit
* MPI functions return `MPI_SUCCESS` on success or an error code (see documentation) on failure depending on how errors are handled.
* By default errors abort (i.e., the error handler is `MPI_ERRORS_ARE_FATAL`).  If you want MPI to return errors for you to handle, you can call [`MPI_Errhandler_set`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Errhandler_set.html)`(MPI_COMM_WORLD, MPI_ERRORS_RETURN)`.
* To abort execution you cannot just `exit` or `return` because there's lots of clean up that needs to be done when running in parallel --- a poorly managed error can easily waste 1000s of hours of computer time.  You must call `MPI_Abort` to exit with an error code.

The new version ([`exercises/mpihello.cc`](https://github.com/wadejong/Summer-School-Materials/blob/master/MPI/exercises/mpihello.cc)) looks like this
```c++
    #include <mpi.h>
    #include <iostream>
    
    int main(int argc, char** argv) {
        MPI_Init(&argc,&argv);
        
        int nproc, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        
        std::cout << "Hello from process " << rank << " of " << nproc << std::endl;
        
        MPI_Finalize();
        return 0;
    }
```

### Compiling MPI programs

Build your parallel version with `make mpihello` or `mpiicpc -o mpihello mpihello.cc`

* MPI provides wrappers for compiling and linking programs because there's a lot of machine specific stuff that must be done.
* For the summer school we are using the GNU stack (`mpicxx`, `mpicc`, `mpifort`, etc.) (see [here](https://software.intel.com/en-us/mpi-developer-reference-linux-compiler-commands) for more details) but if using the Intel C++ compiler and MPI library use the command `mpiicpc` (`mpiifort` for the Intel FORTRAN, `mpiicc` for Intel C). 

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

We used four processes on the local machine (e.g., your laptop or the cluster login node).  More typically, and to avoid the ire of colleagues, we want to use multiple computers from the cluster.  You can manually provide to `mpirun` a hostfile that tells it which computers to use --- on most clusters this is rarely necessary since a batch system is used to
* time share the computers in the cluster
* queue jobs according to priority, resource needs, etc.


Here's an example batch job ([`exercises/mpihello.sbatch`](https://github.com/wadejong/Summer-School-Materials/blob/master/MPI/exercises/mpihello.sbatch)) for SeaWulf annotated so show what is going on (this is for SLURM ---  look at the `*.pbs` version for PBS/Torque):
~~~
     #!/bin/bash
     #SBATCH --nodes=2 --tasks-per-node=40 --cpus-per-task=1 --time=00:05:00 --job-name=test -p debug-40core

     # Above says:
     # - job has 2 (dedicated) nodes with 40 processes per node and one cpu/thread per process with a 5 minute max runtime
     # - use the debug-40core queue
     # - name the job "test"
     # - (slurm by default merges the standard output and error into one file)

     # Output should appear in the file "slurm-<jobnumber>.out" in the
     # directory from which you submitted the job

     # ================================================
     # Slurm by default 
     # * copies your environment variables from when you submitted the job, so if your
     #   modules were correct at time then you don't need to load them here.  If not
     #   you should execute the following command
     #   source /gpfs/projects/molssi/modules-gnu
     # * starts the job running in the same directory that you submitted it from.

     # Uncomment this if you want to see other SLURM environment variables
     #env | grep SLURM

     # Finally, run the executable using #tasks_per_node on each of #nodes
     mpirun ./mpihello

     # You can run more things below or use different numbers of processes
     #mpirun -np 4 ./mpihello
~~~
But I find the comments distracting, so here ([`exercises/mpihello.sbatch`](https://github.com/wadejong/Summer-School-Materials/blob/master/MPI/exercises/mpihello_minimal.sbatch)) is a minimal version.
~~~
     #!/bin/bash
     #SBATCH --nodes=2 --ntasks-pepr-node=40 --cpus-per-task=1 --time=00:05:00 --job-name=test -p debug-40core

     mpirun ./mpihello
~~~
You can copy and edit the file for your other jobs.  Note that other other systems running PBS and other schedulers will differ.

Submit the job from the directory holding your executable (or modify the batch script accordingly)
~~~
    qsub mpihello.pbs
~~~

Useful PBS/Torque commands are
* `qstat` --- see all queued/running jobs
* `qstat -u <username>` --- to see just your jobs
* `qstat -f <jobid>` --- to see detailed info about a job
* `qstat -Q` and `qstat -q` --- to see info about batch queues (for the summer school only `molssi` is available)
* `qdel <jobid>` --- to cancel a job

and for SLURM
* `squeue` --- see all queued/running jobs
* `squeue -u <username>` --- to see just your jobs
* `scontrol show job <jobid>` --- to see detailed info about a job
* `sinfo` --- to see info about batch queues
* `scancel <jobid>` --- to cancel a job


##  4. Sending and receiving messages --- point to point communication

### Essential elements
1. Process rank, message tag, MPI data type, communicator size
2. Blocking communication
3. One minimal set of six operations
3. Buffering and safe communication
4. Non-blocking communication
5. Implied weak synchronization
6. Other communication modes (synchronous send, buffered send)

A process is identified by its rank --- an integer `0,1,..,P-1` where `P` is the number of processes in the communicator (`P` is the size of the communicator).

Every message has a `tag` (integer) that you can use to uniquely identify the message.  This implements the CSP concept of message type --- MPI uses the term `tag` in order to distinguish from the datatype being sent (byte, integer, double, etc.).

In a conversation between a pair of processes, messages of the same `tag` are received in the order sent.

But messages from multiple processes can be interleaved with each other.


### Blocking communication

* When a blocking send function ([`MPI_Send`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Send.html)) completes the buffer can immediately be reused without affecting the sent message.  Note that the receiving process may not necessarily have yet received the message.
* When a blocking recv function ([`MPI_Send`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Recv.html)) completes the received message is fully available in the buffer.

```c++
    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,  MPI_Comm comm);
```
* `buf` --- pointer to the start of the buffer being sent
* `count` --- number of elements to send
* `datatype` --- MPI data type of each element
* `dest` --- rank of destination process
* `tag`  --- message tag
* `comm` --- the communicator to use

```c++
    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
```
* `buf` --- pointer to the start of the buffer to receive the message
* `count` --- maximum number of elements the buffer can hold
* `datatype` --- MPI data type of each element
* `source` --- rank of source process ---  `MPI_ANY_SOURCE` matches any process
* `tag`  --- message tag (integer `>= 0`) --- `MPI_ANY_TAG` matches any tag
* `comm` --- the communicator to use
* `status` --- pointer to the structure in which to store status

The actual source and tag of the received message can be accessed directly from the status.  Call [`MPI_Get_count`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Get_count.html) to get the count.
```c++
    status.MPI_TAG;
    status.MPI_SOURCE;
    MPI_Get_count( &status, datatype, &count );
```

There's data types for everything, and you can define you own including non-contiguous data structures:

|**MPI data type**  |**C data type**     |
|:------------------|:-------------------|
|`MPI_BYTE`           |8 binary digits     |
|`MPI_CHAR`           |char                |
|`MPI_UNSIGNED_CHAR`  |unsigned char       |
|`MPI_SHORT`          |signed short int	 |	 
|`MPI_UNSIGNED_SHORT` |unsigned short int	 |	 
|`MPI_INT`            |signed int          |
|`MPI_UNSIGNED`       |unsigned int	 |	 
|`MPI_LONG`           |signed long int	 |	 
|`MPI_UNSIGNED_LONG`  |unsigned long int	 |	 
|`MPI_FLOAT`          |float               |
|`MPI_DOUBLE`         |double              |
|etc.               |                    |
|`MPI_PACKED`	    |define your own with|
|                   |[`MPI_Pack`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Pack.html)/[`MPI_Unpack`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Pack.html) |


#### Exercise:

Write a program to send an integer (`=99`) from process 0 to process 1 and verify the value is correct.  If it is correct, then print "OK" and terminate correctly, otherwise abort.  Run your program.

Hint: start by copying `exercises/mpihello.cc`.

#### Exercise:

Write a program that has two processes exchanging a buffer of length `N` bytes.  Initialize the buffers to the process rank and verify the exchange happened correctly (i.e., the elements in the buffer received by process 1 should have the value 0).  Try `N=10**n` for `n=0,1,2,3,4,5,6,7,8` (i.e., go from small to very large messages).

There are several ways (both correct and incorrect) of writing this program.  You might try the simplest option first --- each process first sends its buffer and then receives its buffer.

Note that this is such a common operation that there is a special ([`MPI_Sendrecv`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Sendrecv.html)) operation to make this less error prone, less verbose, and to enable optimizations.

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

A timer is also useful --- [`MPI_Wtime`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Wtime.html) returns a high precision wall clock (elapsed) time.  Note that clocks on each process are **not** synchronized.

### Non-blocking (asynchronous) communication

* When a non-blocking send function ([`MPI_Isend`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Isend.html)) completes, the user must not modify the send buffer until the request is known to have completed (e.g., using ([`MPI_Test`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Test.html)) or [`MPI_Wait`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Wait.html)).

* When a non-blocking recv function ([`MPI_Irecv`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Irecv.html)) completes, any message data is not completely available in the buffer until the request is known to have completed.

```c++
    int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,  MPI_Comm comm, MPI_Request *request);
    int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,  int tag, MPI_Comm comm, MPI_Request *request);
```
* `request` --- a pointer to structure that will hold the information and status for the request

```c++
    int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
    int MPI_Wait(MPI_Request *request, MPI_Status *status);
    int MPI_Cancel(MPI_Request *request);
```
* `request` --- a pointer to the request being tested/waited-upon/cancelled
* `flag` --- a pointer to an int that will be non-zero (true) if the operation has completed
* `status` --- a pointer to the structure in which status will be stored if the operation has completed
* See also `MPI_Waitall`, `MPI_Waitany` and `MPI_Waitsome` for waiting on multiple requests


### Other P2P communication modes

*  Buffered send --- provide sender-side buffering to ensure a send always completes and to make memory-management more explicit
*  Synchronous send --- completes on the sender-side when the receive has also completed
*  Ready send --- if you know a matching receive has already been posted this enables optimizations, and this style of programming is explicitly safe from memory/buffer issues
*  One-sided operations --- remote memory access (RMA)


## 5. Global or collective operations

### Essential elements
1. Broadcast
2. Reduction
3. Implied global synchronization
4. Other global operations

In constrast to point-to-point operations that involve just two processes, global operations move data between **all** processes asscociated with a communicator with an implied **synchronization** between them.  All processes within a communicator are required to invoke the operation --- hence the alternative name "collective" operations.

Many chemistry, materials, and biophysics applications are written using only global operations to
* share/replicate information between all processes by broadcasting, and
* compute sums over (partial) results computed by each processes.

This approach does not necessarily scale to the very largest supercomputers, but can suffice for many needs.

We introduce broadcast and reduction, and then work through an example.

### Broadcast

([`MPI_Bcast`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Bcast.html)) broadcasts a buffer of data from process rank `root` to all other processes.  Once the operation is complete within a process its buffer contains the same data as that of process `root`.
```c++
    int MPI_Bcast (void *buffer, int count, MPI_Datatype datatype, int root,  MPI_Comm comm)
```
* `root` --- the process that is broadcasting the data --- this **must** be the same in all processes

### Reduction

To combines values from all processes with a reduction operation either to just process `root` (([`MPI_Reduce`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Reduce.html))) or distributing the result back to all processes (([`MPI_Allreduce`](https://www.mpich.org/static/docs/v3.2/www3/MPI_Allreduce.html))).

```c++
    int MPI_Reduce (const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

    int MPI_Allreduce (const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm );
```
* `sendbuf` --- a pointer to the buffer that contains the local data to be reduced
* `recvbuf` --- a pointer to the buffer that will hold the result

There are many pre-defined reduction operation and you can also define your own

|**Operation** | Description | Datatype|
|:-------------|:------------|:--------|
|`MPI_MAX`       |maximum      |integer,float|
|`MPI_MIN`       |minimum      |integer,float|
|`MPI_SUM`       |sum          |integer,float|
|`MPI_PROD`      |product      |integer,float|
|`MPI_LAND`      |logical AND  |integer|
|`MPI_BAND`      |bit-wise AND |integer,MPI_BYTE|
|`MPI_LOR`       |logical OR   |integer|
|`MPI_BOR`       |bit-wise OR  |integer,MPI_BYTE|
|`MPI_LXOR`      |logical XOR  |integer|
|`MPI_BXOR`      |bit-wise XOR |integer,MPI_BYTE|
|`MPI_MAXLOC`    |max value and location|float|
|`MPI_MINLOC`    |min value and location|float|

### Exercise

In [`exercises/pi_seq.cc`](https://github.com/wadejong/Summer-School-Materials/blob/master/MPI/exercises/pi_seq.cc) is a (now traditional) Monte Carlo program to compute the value of pi.  Make it run in parallel using broadcast and reduce.  Increase the number of points to demonstrate a speedup.

We will walk through the solution together since this is an important example.

### Exercise:

In [`exercises/trapezoid_seq.cc`](https://github.com/wadejong/Summer-School-Materials/blob/master/MPI/exercises/trapezoid_seq.cc) is a sequential program that uses the trapezoid rule to estimate the value of the integral

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\int&#95;{-6}^{6}\exp(-x^2)\cos(3x)\,dx" title="Amdahl" />

It repeatedly increases the number of points by a factor two until the error is satisfactory.

Please make it run in parallel using MPI.  Initially just make the integration step run in parallel using all-reduce. Then, make only process 0 responsible for deciding if the error is satisfactory (and of course telling everyone else).

We will walk through the solution together since this is an important example.

### Other global operations

There are many other global operations --- barrier, gather, scatter, parallel prefix, etc.

There are also asynchronous variants --- which are interesting because they may permit overlap of multiple global operations, or overlap of work and communication, or reduce the impact of load imbalance.


### Another minimal set of six operations

~~~
    MPI_Init
    MPI_Finalize
    MPI_Comm_size
    MPI_Comm_rank
    MPI_Bcast
    MPI_Reduce
~~~
Or a total of eight if you include
~~~
    MPI_Send
    MPI_Recv
~~~
Or 9 if you want a timer
~~~
    MPI_Wtime
~~~

##  6. Reasoning about performance

Essential concepts to reason about the performance of your message passing application

* Amdahl's law, speed up and efficiency

* Weak and strong scaling

* Load balance, data balance, hot spots, synchronizations

* Latency and bandwidth of communication

### Amdahl's law

[Ahdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law) is both simple and brutal.  Dividing the execution time into a sequential component (*Ts*) and perfectly parallel program (*Tp*) the execution time on *P* processes is then

<img src="https://latex.codecogs.com/svg.latex?\Large&space;T(P)=Ts+\frac{Tp}{P}" title="Amdahl" />

Speedup is the ratio of sequential to parallel execution time (whereas efficiency is the ratio of the ideal speedup, *P*, and the actual speedup)

<img src="https://latex.codecogs.com/svg.latex?\Large&space;S(P)=\frac{Ts+Tp}{Ts+\frac{Tp}{P}}" title="speedup" />

You can see that (assuming *Tp>>Ts*) the maximum speed up is *Tp/Ts*.  I.e., to get a speedup of 100, 99% of your work must run perfectly parallel.  To get a speedup of 1,000,000  (the size of modern supercomputers) 99.9999% perfect parallelism is necessary.   Bear this in mind when setting performance goals and expectations.

For these reasons the concepts of strong and weak scaling were introduced.
* Strong scaling: An application shows good strong scaling if for a fixed problem size it shows a near ideal (linear in *P*) speedup as you add more processes.  This is hard because of Amhdahl's law, but unfortunately is what we often want to do in molecular science.

* Weak scaling: An application shows good weak scaling if its execution time remains constant as the amount of work is increased proportional to the number of processors.  I.e., on bigger machines you run bigger problems that ideally have more parallelism.  However, while this is perhaps straightforward for many grid-based engineering code for which the amount of work scales linearly with the amount of data, many chemistry and materials applications have non-linear scaling.

### Load, data balance and hot spots

Further limiting performance is the assumption of perfect parallelism.  It can be very hard to distribute work (a.k.a. load balance) across all of the processes.  For some applications, work is entirely driven by the data but this is not always the case.  A process that has too much work is sometimes referred to as a hot spot (also used to refer to a compute-intensive block of code).

Data distribution can also be a challenge.  The finite memory of each node is one constraint.  Another is that all of the data needed by a task must be brought together for it to be executed.  A non-uniform data distribution can also lead to communication hot spots (processors that must send/recv a lot of data) or hot links (wires in the network that are heavily used to transmit data).  This last point highlights the role of network topology --- the communication pattern of your application is mapped onto the wiring pattern (toplogy) of your network.  A communication intensive application may be sensive to this mapping, and MPI provides some assistance for this (e.g., see [here](http://wgropp.cs.illinois.edu/courses/cs598-s16/lectures/lecture28.pdf)).  See also bisection bandwidth below.

The performance impact of poor load balance will become apparent at synchronization points (e.g., blocking global communications) where all processes must wait for the slowest one to catch up.

### Latency and bandwidth

For point-to-point communication, the central concepts are latency (*L*, the time in seconds for a zero-length message) and bandwidth (*B*, speed of data transfer in bytes/second).  These enable us to model the time to send a message of *N* bytes as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;T(N)=L+\frac{N}{B}" title="LB" />

For typical modern computers *L*=1-10us and *B*=1-100Gbytes/s.  It is hard to accurately measure the latency since on modern hardware the actual cost can depend upon what else is going on in the system and upon your communication pattern.  The bandwidth is a bit easier to measure by sending very large messages, but it can still depend on communication pattern and destination.

An important and easy to remember value is *N1/2*, which is the message length necessary to obtain 50% of peak bandwidth (i.e., *T(N)=2N/B*)

<img src="https://latex.codecogs.com/svg.latex?\Large&space;N&#95;{1/2}=LB" title="Nhalf" />

Inserting *L*=1us and *B*=10Gbyte/s, we obtain *N1/2*=10000bytes.

Excercise: Derive a similar a similar formula for the length necessary to acheive 90% peak bandwith.

Bisection bandwidth is another important concept especially if you are doing a lot of communication all at once.  The network connecting the computers (nodes) in your cluster probably does not directly connect all nodes with every other node --- for anything except a small cluster this would involve too many wires (*O(P^2)* wires, *P*=number of nodes).  Instead, networks use simpler topologies with just *O(P)* wires --- e.g., mesh, n-dimension torous, tree, fat tree, etc.  Imagine dividing this network in two halves so as to give the worst possible bandwidth connecting the halves, which you can derive by counting the number of wires that you cut. This is the bisection bandwidth.  If your application is doing a lot of communication that is not local but is uniform and random, your effective bandwidth is the bisection bandwidth divided by the number of processes.  This can be much smaller than the bandwidth obtained by a single message on a quiet machine.  For instance, in a square mesh of P processes, the bisection bandwidth is *O(sqrt(P))* and if all proceses are trying to communicate globally the average available bandwidth is *O(1/sqrt(P))*, which is clearly not scalable.

Thus, the communication intensity and the communication pattern of your application are both important.  Spreading communication over a larger period of time is a possible optimization.  Another is trying to communicate only to processes that are close in the sense of distance (wires traversed) on the actual network.

For global communication, the details are more complicated because a broadcast or reduction is executed on an MPI-implementation-specific tree of processes that is mapped to the underlying network topology.  However, for long messages an optimized implementation should be able to deliver similar bandwidth to that of the point-to-point communication, with a latency that grows roughly logarithmically with the number of MPI processes.

##  Debugging

There are some powerful visual parallel debuggers that understand MPI, but since these can be expensive we are often left just with GDB. There are variety of ways of using GDB to debug a parallel application:

* Intel MPI and MPICH provide an easy mechanism --- just add `-gdb` to your `mpirun` command.  At this point you are interacting with `gdb` attached to each of your processes.  By default your commands are sent to all processes and output is annoted process. You can control which process you are interacting with using the `z` command.  Some more info is in section 7 of [here](http://physics.princeton.edu/it/cluster/docs/mpich2-user-guide.pdf).

* Other MPI implementations have other mechanisms.  E.g., see here for [OpenMPI debugging](https://www.open-mpi.org/faq/?category=debugging).

* A more portable solution that assumes the MPI processes can create an X-window on your computer is `mpirun -np 2 xterm -e gdb executable` which creates an `xterm` for each process in your application.  This works great for a few processes, but does not scale and it can be complicated to get X-windows to work thru firewalls, etc.

## 7. Work and data distribution strategies

* partitioning the iterations of an outer loop (see [`exercises/trapezoid_seq.cc`](https://github.com/wadejong/Summer-School-Materials/blob/master/MPI/exercises/trapezoid_seq.cc) and parallel version)
* using a counter to distribute the iterations of a nest of loops (see [`exercises/nest_seq.cc`](https://github.com/wadejong/Summer-School-Materials/blob/master/MPI/exercises/nest_seq.cc) and parallel veersion)
* master slave model
* replicated vs. distributed data
* systolic loop
* parallel matrix multiplication ([here](http://www.cs.utexas.edu/~flame/pubs/SUMMA2d3dTOMS.pdf) and [here](https://www3.nd.edu/~zxu2/acms60212-40212/Lec-07-3.pdf)) is not as easy as you might think
* etc.

## 8. Additional concepts and MPI features

* Process groups
* Inter communicators
* Toplogies
* Implementation attributes
* I/O
* etc.

## 9. Exercises

1. [easy] Skim through some of the other tutorials and documentation that have links provided above
2. [easy-medium] Write a program to benchmark the performance of reduce, all-reduce, broadcast as a function of both N and P.  Use N=1,2,4,8,...,1024*1024 doubles. And experiment with processes on the same node and on
different nodes (this means setting #nodes and #ppn correctly in the PBS file).
4. [easy] Parallelize Monte Carlo computation of pi starting from [`exercises/pi_seq.cc`](https://github.com/wadejong/Summer-School-Materials/blob/master/MPI/exercises/pi_seq.cc) using global operations
4. [easy] Work through the other various examples in the `exercises/` directory
5. [medium] Parallelize the recursively adaptive quadrature program [`exercises/recursive_seq.cc`](https://github.com/wadejong/Summer-School-Materials/blob/master/MPI/exercises/recursive_seq.cc)
6. [medium-hard] Write MPI versions of the example SCF, VMC, or MD codes in the main [chemistry examples directory](https://github.com/wadejong/Summer-School-Materials/blob/master/examples).  This tree includes example programs for Hartree Fock, molecular dynamics (already seen in the OpenMP lecture), and variational quantum Monte Carlo.  Sequential, OpenMP, and MPI versions are provided, and the `README` in each directory gives more details.  There's lots of different approaches so don't take our parallel versions as definitive.
    * VMC is the easiest
    * MD is also easy to get started, but harder to get best performance
    * Hartree-Fock is actually not that hard but the complexity of the code and algorithm can obscure this
    * Look at the `README.md` files in the `mpi` sub-directories to get hints.

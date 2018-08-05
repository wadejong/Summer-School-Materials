
# Introduction to message passing with MPI

## 1. Outline

1.  Big picture
1.  Hello world in *just* 20 minutes
1.  Sending messages between processes
1.  Global operations
1.  Communicators 
1.  Reasoning about performance
1.  Debugging, etc.
1.  Additional material
1.  Exercises
1.  Review of exercise solutions
  
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
    * An intra communicator encapsulates all information and resources needed for a group of processes to communicate with each other
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

We used four processes on the local machine (e.g., your laptop or the cluster login node).  More typically, we want to use multiple computers and most clusters use a batch system to
* time share the computers in the cluster
* queue jobs according to priority, resource needs, etc.

**EXAMPLE BATCH JOB HERE**

You can also provide to `mpirun` a hostfile that tells it which computers to use --- on most clusters this is rarely neccessary.


#### Some mpirun options


##  Sending and receiving messages --- point to point communication

A process is identified by its rank --- an integer `0,1,..,P-1` where `P` is the number of processes in the communicator (`P` is the size of the communicator).

Every message has a `tag` (integer) that you can use to uniquely identify the message.  This implements the CSP concept of message type --- MPI uses the term `tag` in order to distinguish from the datatype being sent (byte, integer, double, etc.).

In a conversation between a pair of processes, messages of the same `tag` are received in the order sent.

But messages from multiple processes can be interleaved with each other.


#### Blocking communication

* When a blocking send operation completes the buffer can immediately be reused without affecting the sent message.  Note that the receiving process may not necessarily have yet received the message.
* When a blocking recv operation completes the received message is fully available in the buffer.

~~~
    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,  MPI_Comm comm)
~~~
* `buf` --- pointer to the start of the buffer being sent
* `count` --- number of elements to send
* `datatype` --- MPI data type of each element
* `dest` --- rank of destination process
* `tag`  --- message tag
* `comm` --- the communicator to use

~~~
    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
~~~
* `buf` --- pointer to the start of the buffer to recieve the message
* `count` --- maximum number of elements the buffer can hold
* `datatype` --- MPI data type of each element
* `source` --- rank of source process ---  `MPI_ANY_SOURCE` matches any process
* `tag`  --- message tag (integer `>= 0`) --- `MPI_ANY_TAG` matches any tag
* `comm` --- the communicator to use
* `status` --- pointer to the structure in which to store status

The actual source, count and tag of the received message can be accessed from the status.

~~~
    status.MPI_TAG;
    status.MPI_SOURCE;
    MPI_Get_count( &status, datatype, &count );
~~~


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

Take 5 minutes 





### non-blocking (asynchronous) communication




Essential concepts:
*  Blocking send/receive
*  Non-blocking send/receive

Less essential:
*  Buffered send
*  Synchronous send
*  





Sometimes it is confusing where stuff is actually running, in part because every job scheduler and resource manager seems to be set up differently, so we modify the program to also print out the name of the machine each process is running on



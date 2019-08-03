
# Introduction to SIMD vector programming

## 1. Outline

1.  Big picture
1.  Quick review of program execution
1.  Quick review of pipelining
1.  Quick review of SIMD with focus on x86
1.  Useful links

For this section we will be using the latest Intel Compiler so please execute this command in your shell to load the right modules
~~~
   source /gpfs/projects/molssi/modules-intel
~~~
The latest GNU compiler is actually also pretty good at vectorization, but unfortunately it also needs an up-to-date GLIBC and the one currently installed on Seawulf is too old.

### 1.1 Useful links



* [Intel® Architecture Instruction Set Extensions and Future Features Programming Reference](https://software.intel.com/sites/default/files/managed/c5/15/architecture-instruction-set-extensions-programming-reference.pdf)

* 



## 2. Big picture

In this session we will be focusing on how to extract peak performance (i.e., maximum parallelism) from a single modern processor core.  Tomorrow we will look at using multiple cores (via OpenMP) and multiple machines (via MPI).  We will focus on Intel x86, but the ideas will apply to essentially all modern processors (e.g., Intel/AMD x86, IBM Power, ARM v8 NEON or SVE, ...) including even GPGPUs.

There can be a factor of 128 or greater between the performance of serial code and fully optimized code running on a *single* core.  So whether your are designing a new code or tuning an existing code you cannot ignore single core performance.  Similarly, when comparing performance benchmarks on different architectures you must be careful to inquire about what optimizations were performed.

Nearly all of the recent x86 architectural enhancements relating to HPC or data-intensive applications have come from enhanced vectorization and specialized functional units.

Key elements of modern CPU were architecture already covered in the introduction
* multi-issue instruction architecture
* pipelining
* scalar and SIMD registers
* cache
* memory
* multiple cores

and now we put your understanding into practice.

### 5. Quick review of program execution

There are multiple functional units, e.g.,
* integer arithmetic and adress computation
* floating point arithmetic
* memory read
* memory write
* etc.

and in most processors it is usually possible in a single clock cycle to issue an instruction to 

Instructions are read from memory, decoded, and the execution engine (with dependency analysis and possibly speculative look ahead) tries to bundle as many instructions for each independent functional units as possible for issue each clock cycle.

### 3. Quick review of pipelining

A complex instruction may take multiple cycles to complete --- the *latency* (*L*).

Pipelining tries to hides this latency so that you can get a result every clock cycle instead of every *L* clock cycles --- a factor of *L* speedup.

This is accomplished by dividing the operation into multiple stages, one per cycle, and overlapping stages of performing successive operations

E.g., floating point multiplication with a 3 stage pipeline (we'll ignore the necessary memory accesses and just imagine the data is already loaded into the registers)
~~~
    for (i=0; i<6; i++) a[i]*b[i];
~~~
| cycle | stage0 | stage1 | stage2 | result |
| ------| ------ | ------ | ------ | ------ |
|  0    | a0*b0  |        |        |        |
|  1    | a1*b1  | a0*b0  |        |        |
|  2    | a2*b2  | a1*b1  | a0*b0  |        |
|  3    | a3*b3  | a2*b2  | a1*b1  | a0*b0  |
|  4    | a4*b4  | a3*b3  | a2*b2  | a1*b1  |
|  5    | a5*b5  | a4*b4  | a3*b3  | a2*b2  |
|  6    |        | a5*b5  | a4*b4  | a3*b3  |
|  7    |        |        | a5*b5  | a4*b4  |
|  8    |        |        |        | a5*b5  |

The first result takes $L=3$ cycles to appear, but after that we get one result per clock cycle.  Thus, the execution time (cycles) is
~~~
    T = L + n - 1
~~~
Note that there are some empty stages while the pipleline is filling up and draining, so our efficiency is not 100% unless we have very large *n*

#### Exercise: how big must *n* be to reach 50% of peak performance --- this is Hockney's <a href="https://www.codecogs.com/eqnedit.php?latex=n_{1/2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{1/2}" title="n_{1/2}" /></a>.  

The speed (operations per cycle) is *n/T  = n / (L+n-1)*.  The peak speed is 1 op/cycle, so 50% of peak speed is *1/2*.  Solving <a href="https://www.codecogs.com/eqnedit.php?latex=n_{1/2}&space;=&space;L-1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{1/2}&space;=&space;L-1" title="n_{1/2} = L-1" /></a>.  Verify from the table that 50% of peak speed is obtained with *n=2*.

What about for 90% of peak speed?  <a href="https://www.codecogs.com/eqnedit.php?latex=n_{90\%}&space;=&space;9&space;(L-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{90\%}&space;=&space;9&space;(L-1)" title="n_{90\%} = 9 (L-1)" /></a>.  For our example, we will need a vector length of 9*2=18 to reach 90% of peak speed.

What about for 99% of peak speed?  <a href="https://www.codecogs.com/eqnedit.php?latex=n_{99\%}&space;=&space;99&space;(L-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{99\%}&space;=&space;99&space;(L-1)" title="n_{99\%} = 99 (L-1)" /></a>


### 4. Quick review of SIMD with focus on x86

Instruction decode is expensive in chip area and power, and moving data from multiple registers to multiple functional units is similarly expensive.  By having a single instruction operate on multiple data (SIMD) we simplfy both instruction decode and data motion.  

x86 register names:
* xmm --- SSE 128-bit register (16 bytes, 8 shorts, 4 ints or floats, 2 doubles)
* ymm --- AVX 256-bit register (32 bytes, ..., 4 doubles)
* zmm --- AVX512 512-bit register (64 bytes, ..., 8 doubles)

By operating on all *W* (width) elements simultaneously you get a factor of *W* speedup.

An element in a vector register is often referred to as a lane (think of vector processing as parallel lanes of traffic moving lock step together). 

A SIMD instruction operates on all elements in a register.  E.g., *a\*b*
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}a_0\\a_1\\a_2\\a_3\end{pmatrix}&space;*&space;\begin{pmatrix}b_0\\b_1\\b_2\\b_3\end{pmatrix}&space;\rightarrow&space;\begin{pmatrix}a_0&space;*&space;b_0\\a_1*b_1\\a_2*b_2\\a_3*b_3\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}a_0\\a_1\\a_2\\a_3\end{pmatrix}&space;*&space;\begin{pmatrix}b_0\\b_1\\b_2\\b_3\end{pmatrix}&space;\rightarrow&space;\begin{pmatrix}a_0&space;*&space;b_0\\a_1*b_1\\a_2*b_2\\a_3*b_3\end{pmatrix}" title="\begin{pmatrix}a_0\\a_1\\a_2\\a_3\end{pmatrix} * \begin{pmatrix}b_0\\b_1\\b_2\\b_3\end{pmatrix} \rightarrow \begin{pmatrix}a_0 * b_0\\a_1*b_1\\a_2*b_2\\a_3*b_3\end{pmatrix}" /></a>

Modern AVX transformed the ease of obtaining high peformance
* wider registers
* more registers
* many more data types supported
* gather+scatter (operate on non-contiguous data)
* relaxed previous constraints on aligning data in memory
* better support for reductions across a register
* many more operations including math functions (sin, cos, ...)
* fully predicated (historically a key advantage of NVIDIA GPUs over x86 SIMD)

### 4.1 SIMD on a long vector

If the SIMD register is just *W* words wide, how do we operate on a vector of arbitrary length *N*?  The loop is tiled
~~~
   for (i=0;i<n;i++) a[i] = b[i]*10.0 + c[i];
~~~
becomes
~~~
   NR = n/W;
   for (R=0; R<NR; R++,i+=W)
      a[i:i+W] = b[i:i+W]*10.0 + c[i:i+W]; // vector op using Python slice notation
   
   for (;i<n;i++) a[i] = b[i]*10.0 + c[i]; // clean up loop
~~~
In practice, things can be more complex due to handling address misalignment, unrolling, etc.


### 4.2 Predication

There are lots of reasons for not wanting to operate (or write or read) on all elements in a SIMD register.  E.g., perhaps your vector is shorter than the register, or perhaps you have some test that must be true for data that you want to compute on (also termed computing under a mask).

E.g., consider this loop
~~~
   for (i=0;i<n;i++) if (a[i]<10) a[i] = b[i]*10.0 + c[i];
~~~

A modern SIMD processor would compute a vector (mask) of boolean (bit) flags 
~~~
   for (i=0;i<n;i++) mask[i] = (a[i]<10);
~~~
and the actual computation becomes a single predicated vector operation
~~~
   FMA(mask, b, 10, c, a) --- where mask is true compute b*10+c and put result in a
~~~

Before we had predicated vector instructions, the compiler had to do a lot more work to produce vector code which was also much slower.  As a result, lots of loops were just marked unvectorizable either because the compiler could not figure it out or it did not seem worthwhile.  Nowadays, most simple predicates are vectorizable.

Examples of AVX512 operations:
|Operation and souces/targets             | Description                                             |
|-----------------------------------------|---------------------------------------------------------|
|VADDPD zmm1 {k1}, zmm2,zmm3/m512/m64bcst |  Add packed double-precision floating-point values from |
|                                         |  zmm3/m512/m64bcst to zmm2 and store result in zmm1     |
|                                         |  with writemask k1                                      |
|-----------------------------------------|---------------------------------------------------------|
|VFMADD213PD zmm1 {k1},zmm2,zmm3/m512/m64bcst |  Multiply packed double-precision floating-point values from |
|                                             |  zmm1 and zmm2, add to zmm3/m512/m64bcst and put result in zmm1 |
|                                             |  with write mask k1                                 |
|-----------------------------------------|---------------------------------------------------------|


Fortunately, we humans no longer need to write in assembly language.


### 4.3 Memory alignment

Historically, a load into vector register of width *W* bytes could only be performed from memory addresses that were appropriately aligned (usually on an address that was a multiple of *W*).  Subsequently, one could perform unaligned loads but only with a severe performance penalty.  Most recently, unaligned loads suffer a much lower peformance impact, but there can still benefit from aligning data.

For best performance, try to align data structures and pad leading dimensions of multi-dimension structures accordingly.  Since this is not a big concern on modern x86 we won't dwell upon it.
* Declare C++ variables using `alignas(W)`
* Allocate on heap with `posix_memalign` or `std::aligned_alloc` (C++ 17)
* Multi-dimension array `a[10][7]` perhaps best stored as `a[10][8]` for AVX

### 4.4 Pipelined SIMD

A factor of *L\*W* speedup compared to serial code.

### 4.5 Exercises

#### Exercise: what is the peak spead (operations/element/cycle) of a single, piplelined, SIMD functional unit with width *W=8* and latency *L=3*? 8.

#### Exercise: how long must your vector be to obtain 90% of peak speed from a single, piplelined, SIMD functional unit with width *W=8* and latency *L=3*?  8*9*(3-1) = 144.   

#### Exercise: what is the peak floating performance of a single core of sn-mem, and what do you have to do to get it?


Login into the Seawulf node `sn-mem` --- if you are already logged into a login node just do `ssh sn-mem`, otherwise from your laptop do
~~~
    ssh username@sn-mem.seawulf.stonybrook.edu
~~~
[Note that only the `login`, `sn-mem` and `cn-mem` nodes are directly accessible from the outside]


What is the processor and how many cores are there on the machine?  `cat /proc/cpuinfo`

~~~
processor	: 0
...
...
...
processor	: 143
vendor_id	: GenuineIntel
cpu family	: 6
model		: 85
model name	: Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz
...
flags		: ... sse sse2 ssse3 fma sse4_1 sse4_2 avx avx2 avx512f avx512dq avx512cd avx512bw avx512vl ...
...
~~~

Each physical x86 core can support two (hyper)threads and each of these appears to Linux as a core.  Thus, there are 72 physical cores on the system.

Googling for "Intel Xeon Gold 6154" will find [this page](https://ark.intel.com/content/www/us/en/ark/products/120495/intel-xeon-gold-6154-processor-24-75m-cache-3-00-ghz.html). Which tells you this is a Skylake processor with 18 physical cores per socket, so `sn-mem` has 4 sockets.

The Wikipedia page on [Skylake](https://en.wikipedia.org/wiki/Skylake_(microarchitecture)) is not especially helpful.  For the full technical details (much more than most of us need)
* [Intel 64 and IA-32 Architectures Optimization Reference Manual](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf)
* Agner Fog [The microarchitecture of Intel, AMD and VIA CPUs](https://www.agner.org/optimize/microarchitecture.pdf) --- section 11, p149, deals with Skylake
* Agner Fog [Instruction tables](https://www.agner.org/optimize/instruction_tables.pdf)
* If you are interested in detail about the memory performance, look [here](https://www.anandtech.com/show/11544/intel-skylake-ep-vs-amd-epyc-7000-cpu-battle-of-the-decade/12).

Key points --- it has
* AVX512, and
* a throughput of two 512-bit operations per clock cycle including fused-multiply-add.

Thus, the peak speed of a single core is 96 GFLOP/s in double precision (or 192 GFLOP/s in single precision)
~~~
    frequency * SIMD width * instruction throughput * ops/instruction  =  peak Gops/sec
       3.0    *    8       *          2             *      2           =     96
~~~
and to get this performance you must issue 2 512-bit FMA instructions every single cycle.


[Aside: hence, the peak speed of the entire node is 72*96 GFLOP/s = 6.9 TFLOP/s, and for comparison the attached NVIDIA P100 is 4.7 TFLOP/s.]


#### Exercise: repeat the analysis for the Seawulf login node


#### Exercise: repeat the analysis for your laptop


## 5.0 Benchmarking DAXPY

Measure the cycles/iteration of this loop as a function of `n`
~~~
  for (int i=0; i<n; i++) {
      y[i] = a*x[i] + y[i];
  }
~~~
or, equivalently, from the Intel Vector Math Library (VML)
~~~
  cblas_daxpy (n, a, x, 1, y, 1);
~~~

Before you run test --- what do you expect to see?

On `sn-mem` compile and run in [`Examples/Vectorization/bench`](https://github.com/wadejong/Summer-School-Materials/blob/master/Examples/Vectorization/bench)

### 5.1 Observations

In the figure are cycles/element for DAXY as a function of N (the x-axis being on a log scale)

![measured](https://github.com/wadejong/Summer-School-Materials/blob/master/VectorSIMD/plot.gif  "DAXPY cycles/element")

### 5.2 Analysis

Theoretical peak speed on Skylake --- in one cycle can issue 
* FMA AVX512 SIMD
* 2 loads AVX512 SIMD
* 1 store AVX512 SIMD
* loop counter increment
* test and branch

The L1 cache is 32KB or 2048*64bytes (we have 2 d.p. vectors, so this is the number of elements that can fit into L1 assuming 

and the L1 cache can, at least in theory, support full-speed memory access.

So the entire loop can be written to execute in just one cycle! This would be a throughput of 0.125 cyles/element.  CBLAS best is 0.163, likely due to the L1 cache not being able to sustain the peak speed due to bank conflicts.

[Aside: nice detailed analysis of Haswell cache access in the comments [here](https://software.intel.com/en-us/forums/intel-moderncode-for-parallel-architectures/topic/608964).]



#### Exercise: change the loop stride from 1 to 2, 3, 4, 8, 12, 16, 256




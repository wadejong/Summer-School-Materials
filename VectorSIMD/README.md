
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


## 2. Big picture

In this session we will be focusing on how to extract peak performance (i.e., maximum parallelism) from a single modern processor core.  Tomorrow we will look at using multiple cores (via OpenMP) and multiple machines (via MPI).  We will focus on Intel x86, but the ideas will apply to essentially all modern processors (e.g., Intel/AMD x86, IBM Power, ARM v8 NEON or SVE, ...) including even GPGPUs.

There can be a factor of 128 or greater between the performance of serial code and fully optimized code running on a *single* core.  So whether your are designing a new code or tuning an existing code you cannot ignore single core performance.  Similarly, when comparing performance benchmarks on different architectures you must be careful to inquire about what optimizations were performed.

Nearly all of the recent x86 architectural enhancements relating to HPC or data-intensive applications have come from enhanced vectorization and specialized functional units.

Key elements of modern CPU were architecture already covered in the introduction
* multi-issue instructions
* SIMD instructions
* pipelining
* registers
* cache
* memory
* multiple cores

and now we put your understanding into practice.

### 5. Quick review of program execution

There are multiple function units, e.g.,
* integer arithmetic and adress computation
* floating point arithmetic
* memory read
* memory write
* etc.

and in most processors it is usually possible in a single clock cycle to issue an instruction to 

Instructions are read from the executable, decoded, and the execution engine (often with possibly speculative look ahead) tries to bundle as many instructions to independent function units as possible for issue each clock cycle.



### 3. Quick review of pipelining

A complex instruction may take multiple cycles to complete --- the *latency* (*L*).

Pipelining tries hides this latency so that you can get a result every clock cycle instead of every *L* clock cycles.  This is accomplished by dividing the operation into multiple stages, one per cycle, and overlapping stages of performing successive operations

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

The speed (operations per cycle) is *n/T  = n / (L+n-1)*.  The peak speed is 1 op/cycle, so 50% of peak speed is *1/2*.  Solving <a href="https://www.codecogs.com/eqnedit.php?latex=n_{1/2}&space;=&space;L-1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{1/2}&space;=&space;L-1" title="n_{1/2} = L-1" /></a>.

What about for 90% of peak speed?  <a href="https://www.codecogs.com/eqnedit.php?latex=n_{90\%}&space;=&space;9&space;(L-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{90\%}&space;=&space;9&space;(L-1)" title="n_{90\%} = 9 (L-1)" /></a>

What about for 99% of peak speed?  <a href="https://www.codecogs.com/eqnedit.php?latex=n_{99\%}&space;=&space;99&space;(L-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{99\%}&space;=&space;99&space;(L-1)" title="n_{99\%} = 99 (L-1)" /></a>


### 4. Quick review of SIMD with focus on x86

Instruction decode is expensive in chip area and power, and moving data from multiple registers to multiple functional units is similarly expensive.  By having a single instruction operate on multiple data (SIMD) we simplfy both instruction decode and data motion.  





#### Exercise: how long must your vector be to obtain 90% of peak speed from a single, piplelined, SIMD functional unit with width *W* and latency *L*?

  


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
and to get this performance we must issue 2 512-bit FMA instructions every single cycle.


[Aside: hence, the peak speed of the entire node is 72*96 GFLOP/s = 6.9 TFLOP/s, and for comparison the attached NVIDIA P100 is 4.7 TFLOP/s.]


#### Exercise: repeat the analysis for the Seawulf login node


#### Exercise: repeat the analysis for your laptop













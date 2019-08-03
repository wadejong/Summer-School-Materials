
# Introduction to SIMD vector programming

## 1. Outline

1.  Big picture
1.  Quick review of pipelining
1.  Quick review of SIMD with focus on x86
1.  Quick review of program execution
1.  Useful links

For this section we will be using the latest Intel Compiler so please execute this command in your shell to load the right modules
~~~
   source /gpfs/projects/molssi/modules-intel
~~~
The latest GNU compiler is actually also pretty good at vectorization, but unfortunately it also needs an up-to-date GLIBC and the one currently installed on Seawulf is too old.


## 2. Big picture

In this session we will be focusing on how to extract peak performance (i.e., maximum parallelism) from a single modern processor core.  Tomorrow we will look at using multiple cores (via OpenMP) and multiple machines (via MPI).  We will focus on Intel x86, but the ideas will apply to essentially all modern processors (e.g., Intel/AMD x86, IBM Power, ARM v8 NEON or SVE, ...) including even GPGPUs.

There can be a factor of 128 or greater between the performance of serial code and fully optimized code running on a *single* core.  So whether your are designing a new code or tuning an existing code you cannot ignore single core performance.  Similarly, when comparing performance benchmarks on different architectures you must be careful to inquire about what optimizations were performed.

Key elements of modern CPU were architecture already covered in the introduction
* multi-issue instructions
* SIMD instructions
* pipelining
* registers
* cache
* memory
* multiple cores

and now we put your understanding into practice.

### 3. Quick review of pipelining


### 4. Quick review of SIMD with focus on x86


### 5. Quick review of program execution



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













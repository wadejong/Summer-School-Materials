
# Introduction to SIMD vector programming

1.  Big picture
1.  Useful links
1.  Auto-vectorization quick start
1.  Quick review of program execution
1.  Quick review of pipelining
1.  SIMD with focus on x86
1.  Peak speed of Intel Skylake (`sn-mem`)
1.  Bencharking DAXPY
1.  Non-trivial example --- vectorizing Metropolis Monte Carlo
1.  Exercises

For this section we will be using the latest Intel Compiler so please execute this command in your shell on Seawulf to load the right modules
~~~
   source /gpfs/projects/molssi/modules-intel
~~~
The latest GNU compiler is actually also pretty good at vectorization on modern Intel processors, but unfortunately it also needs an up-to-date GLIBC and the one currently installed on Seawulf is too old.

## 1. Big picture

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

There are several major techniques for vector/SIMD programming
* Auto-vectorization --- focus of this class
* OpenMP SIMD pragmas --- not covered in detail here, but highly relevant if you are also using OpenMP for threading.
* Use of (vendor-provided) optimized libraries --- always a good idea (note Intel MKL is now free), though compiled code is now often nearly/just-as good especially for short vectors.  Unless the library gives a big performance boost, you might still prefer compiled code since it might kep the code more maintainble and portable.
* SIMD vector intrinsics (assembly-level programming with some help from C++) --- not covered here and not encouraged unless you are really after peak speed
* Assembly programming --- no need these days

## 2. Useful links

Intel compiler
* [Vectorization Essentials](https://software.intel.com/en-us/articles/vectorization-essential) --- recommended reading
* C++ Developer Guide on [Using Automatic Vectorization](https://software.intel.com/en-us/cpp-compiler-developer-guide-and-reference-using-automatic-vectorization)

Intel MKL
* [Getting Started](https://software.intel.com/en-us/get-started-with-mkl-for-linux)
* [Developer Guide](https://software.intel.com/en-us/mkl-linux-developer-guide)

GNU compiler
* [Auto-vectorization in GCC](https://gcc.gnu.org/projects/tree-ssa/vectorization.html) --- a bit out of date
* [GCC Auto-vectorization](http://hpac.rwth-aachen.de/teaching/sem-accg-16/slides/08.Schmitz-GGC_Autovec.pdf)
* Tutorial [SIMD Programming](https://www.moreno.marzolla.name/teaching/HPC/L08-SIMD.pdf)

Clang/LLVM
* [Auto-Vectorization in LLVM](https://llvm.org/docs/Vectorizers.html)

OpenMP SIMD
*  [OpenMP: Vectorization and `#pragma omp simd`](http://hpac.rwth-aachen.de/teaching/pp-16/material/08.OpenMP-4.pdf)

More tutorials
* [SIMD Programming](https://www.eidos.ic.i.u-tokyo.ac.jp/~tau/lecture/parallel_distributed/2018/slides/pdf/simd2.pdf)
* [SIMD Peak FLOPS](https://www.eidos.ic.i.u-tokyo.ac.jp/~tau/lecture/parallel_distributed/2014/slides/simd.pdf)
* [Vectorization and Parallelization of Loops in C/C++ Code](https://pdfs.semanticscholar.org/852c/0115d6011b6cd2746d18f56d64a53e65af5d.pdf)

Lots of gory details (more than most of us need)
* [Intel 64 and IA-32 Architectures Optimization Reference Manual](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf)
* [Intel® Architecture Instruction Set Extensions and Future Features Programming Reference](https://software.intel.com/sites/default/files/managed/c5/15/architecture-instruction-set-extensions-programming-reference.pdf)
* Agner Fog [The microarchitecture of Intel, AMD and VIA CPUs](https://www.agner.org/optimize/microarchitecture.pdf) --- section 11, p149, deals with Skylake
* Agner Fog [Instruction tables](https://www.agner.org/optimize/instruction_tables.pdf)
* Details about Skylake memory performance [here](https://www.anandtech.com/show/11544/intel-skylake-ep-vs-amd-epyc-7000-cpu-battle-of-the-decade/12) or [here](https://www.7-cpu.com/cpu/Skylake.html)

## 3. Auto-vectorization quick start

On `sn-mem` compile (using `make sum`) and run in [`Examples/Vectorization`](https://github.com/wadejong/Summer-School-Materials/blob/master/Examples/Vectorization) the `sum.cc` program.
~~~
#include <iostream>

int main() {
    const int N=100007;
    double a[N];
    for (int i=0; i<N; i++) a[i] = i;
    double sum = 0.0;
    for (int i=0; i<N; i++) sum += a[i];
    std::cout << sum << std::endl;
    return 0;
}
~~~
It adds up the integers 0-100,006 and prints out the sum.  

What are the compiler flags doing?
* `-xHOST` --- optimize for the machine on which you are compiling.  Clearly not a good idea if you will run on a different architecture.
* `-O3` --- enable all level 3 optimizations (level 2 and above include vectorization)
* `-ipo` --- perform inter-procedural analysis including enabling inlining routines between source files (not needed for this example)
* `-no-prec-div -fp-model fast=2` --- relax the floating point accuracy requirements so that some optimizations become permissible (e.g., reordering operations can give different results due to different rounding errors)
* `-qopt-zmm-usage=high` --- force use of the Skylake AVX512 instructions (see below), here, primarily for pedagogical purposes
* `-qopt-report=5 -qopt-report-phase=vec` --- print detailed info about vectorization optimizations into the file  `ipo_out.optrpt` (if the `-ipo` flag is not present the output goes into `<filename>.optrpt`)

Let's look at the optimization report.
~~~
Intel(R) Advisor can now assist with vectorization and show optimization
  report messages with your source code.
See "https://software.intel.com/en-us/intel-advisor-xe" for details.

Intel(R) C++ Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 19.0.4.243 Build 20190416

Compiler options: -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopt-zmm-usage=high -qopt-report=5 -qopt-report-phase=vec -o sum

Begin optimization report for: main()

    Report from: Vector optimizations [vec]


LOOP BEGIN at sum.cc(7,5)
   remark #15388: vectorization support: reference a[i] has aligned access   [ sum.cc(7,29) ]
   remark #15388: vectorization support: reference a[i] has aligned access   [ sum.cc(9,36) ]
   remark #15305: vectorization support: vector length 16
   remark #15399: vectorization support: unroll factor set to 2
   remark #15309: vectorization support: normalized vectorization overhead 0.778
   remark #15301: FUSED LOOP WAS VECTORIZED
   remark #15448: unmasked aligned unit stride loads: 1 
   remark #15449: unmasked aligned unit stride stores: 1 
   remark #15475: --- begin vector cost summary ---
   remark #15476: scalar cost: 9 
   remark #15477: vector cost: 1.120 
   remark #15478: estimated potential speedup: 7.990 
   remark #15487: type converts: 1 
   remark #15488: --- end vector cost summary ---
LOOP END

LOOP BEGIN at sum.cc(9,5)
LOOP END

LOOP BEGIN at sum.cc(7,5)
<Remainder loop for vectorization>
   remark #15388: vectorization support: reference a[i] has aligned access   [ sum.cc(7,29) ]
   remark #15388: vectorization support: reference a[i] has aligned access   [ sum.cc(9,36) ]
   remark #15335: remainder loop was not vectorized: vectorization possible but seems inefficient. Use vector always directive or -vec-threshold0 to override 
   remark #15305: vectorization support: vector length 8
   remark #15427: loop was completely unrolled
   remark #15448: unmasked aligned unit stride loads: 1 
   remark #15449: unmasked aligned unit stride stores: 1 
   remark #15475: --- begin vector cost summary ---
   remark #15476: scalar cost: 9 
   remark #15477: vector cost: 1.120 
   remark #15478: estimated potential speedup: 7.990 
   remark #15487: type converts: 1 
   remark #15488: --- end vector cost summary ---
LOOP END
===========================================================================
~~~

There's a more complicated version in [`sum_timed.cc`](https://github.com/wadejong/Summer-School-Materials/blob/master/Examples/Vectorization/sum_timed.cc) that tries to measure the cost as the number of cycles per element.  

**Exercise:**
1. Make and run the `sum_timed` program
2. Make clean,  add `-no-vec` (to disable vectorization) to the CXXFLAGS, make, and re-run

I got 
* vectorized: 0.0625 --- 16 elements per cycle!  This is as fast as it gets on `sn-mem` (see discussion below about DAXPY)
* un-vectorized: 1.625 --- 26x slower.  The `-no-vec` flag must have done some damage beyond just stopping vectorization.

## 3.1 Requirements/recommendations for vectorizable loops

1. The number of iterations must be computable before the loop starts --- i.e., you cannot modify the iteration count within the loop and you cannot exit the loop prematurely.

2. Contiguous memory access (stride 1) --- non-unit stride access will be slow (since memory read/write are done on entire 64 byte cache lines, wasting bandwidth if you don't use the data).  Indexed read (gather) can inhibit vectorization and is in anycase slow unless most indices are nearly in order, and indexed write (scatter) will inhibit vectorization due to the write dependency.  
   * If there is a lot of computation, you can gather the data into a contiguous array in one loop, compute in a separate loop, and scatter the result in a final loop.
   * The Intel MKL Vector Math Library [Pack/Unpack Functions](https://software.intel.com/en-us/mkl-developer-reference-c-vm-pack-unpack-functions) can accelerate converting strided/indexed/masked vectors to/from contiguous arrays.

3. Aliasing can inibit vectorization --- i.e., the compiler cannot figure out if pointers/arrays refer to non-overlapping memory regions
   * a modern compiler will sometimes generate both scalar and vector code and test at runtime for aliasing
   * you can add the `restrict` keyword to points to assert there is no aliasing
   * also see `pragma ivdep` below

4. Data dependencies inhbit vectorization
   * a variable/array element is written by one iteration and read by subsquent iteration, e.g.,
~~~
    for (i=1; i<n; i++) a[i] = a[i+K];
~~~
If `K>0` then there is no read after write, however, if `K<0` the loop cannot be vectorized.
   * a reduction operation --- should be vectorizable, however, all but the simplest loops seem to confuse some compilers so you may need to use a pragma to indicate the reduction variable.  E.g.,
~~~
#pragma simd reduction(+: sum)
        for (int i=0; i<N; i++) {
            if (vpnew[i] > r[i]*p[i]) {
                x[i] =-vxnew[i];
                p[i] = vpnew[i];
            }
            sum += x[i];
        }
~~~
(for recent versions of GCC and Intel compilers prefer to use `#pragma omp simd reduction(+:sum)` and enable OpenMP on the command line).

If the compiler thinks there is a dependency, but you are confident there is not, you can insert a pragma before the loop.  E.g.,
~~~
#pragma ivdep
    for (i=1; i<n; i++) a[i] = a[i+K];
~~~
(for GCC this is `#pragma GCC ivdep` and in OpenMP `#pragma omp simd`). *Note:* if there really is a dependency, then you just introduced a bug!


## 4. Quick review of program execution

There are multiple functional units, e.g.,
* integer arithmetic and address computation
* floating point arithmetic
* memory read
* memory write
* etc.

Instructions are read from memory, decoded, and the execution engine (with dependency analysis and possibly speculative look ahead) tries to bundle as many instructions as possible for issue each clock cycle, targetting independent functional units.

## 5. Quick review of pipelining

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

The first result takes *L=3* cycles to appear, but after that we get one result per clock cycle.  Thus, the execution time (cycles) is
~~~
    T = L + n - 1
~~~
Note that there are some empty stages while the pipeline is filling up and draining, so our efficiency is not 100% unless we have very large *n*

**Exercise:** How big must *n* be to reach 50% of peak performance --- this is Hockney's <a href="https://www.codecogs.com/eqnedit.php?latex=n_{1/2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{1/2}" title="n_{1/2}" /></a>.  

The speed (operations per cycle) is *n/T  = n / (L+n-1)*.  The peak speed is 1 op/cycle, so 50% of peak speed is *1/2*.  Solving gives <a href="https://www.codecogs.com/eqnedit.php?latex=n_{1/2}&space;=&space;L-1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{1/2}&space;=&space;L-1" title="n_{1/2} = L-1" /></a>.  Verify from the table that 50% of peak speed is obtained with *n=2*.

What about for 90% of peak speed?  <a href="https://www.codecogs.com/eqnedit.php?latex=n_{90\%}&space;=&space;9&space;(L-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{90\%}&space;=&space;9&space;(L-1)" title="n_{90\%} = 9 (L-1)" /></a>.  For our example, we will need a vector length of 9*2=18 to reach 90% of peak speed.

What about for 99% of peak speed?  <a href="https://www.codecogs.com/eqnedit.php?latex=n_{99\%}&space;=&space;99&space;(L-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{99\%}&space;=&space;99&space;(L-1)" title="n_{99\%} = 99 (L-1)" /></a>


## 6. SIMD with focus on x86

Instruction decode is expensive in chip area and power, and moving data from multiple registers to multiple functional units is similarly expensive.  By having a single instruction operate on multiple data (SIMD) we simplify both instruction decode and data motion.  

x86 register names:
* xmm --- SSE 128-bit register (16 bytes, 8 shorts, 4 ints or floats, 2 doubles)
* ymm --- AVX 256-bit register (32 bytes, ..., 4 doubles)
* zmm --- AVX512 512-bit register (64 bytes, ..., 8 doubles)

By operating on all *W* (width) elements simultaneously you get a factor of *W* speedup.

An element in a vector register is often referred to as a lane (think of vector processing as parallel lanes of traffic moving lock step together). 

A SIMD instruction operates on all elements in a register.  E.g., *a\*b* --- elementwise multiplication of two vectors

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}a_0\\a_1\\a_2\\a_3\end{pmatrix}&space;*&space;\begin{pmatrix}b_0\\b_1\\b_2\\b_3\end{pmatrix}&space;\rightarrow&space;\begin{pmatrix}a_0&space;*&space;b_0\\a_1*b_1\\a_2*b_2\\a_3*b_3\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}a_0\\a_1\\a_2\\a_3\end{pmatrix}&space;*&space;\begin{pmatrix}b_0\\b_1\\b_2\\b_3\end{pmatrix}&space;\rightarrow&space;\begin{pmatrix}a_0&space;*&space;b_0\\a_1*b_1\\a_2*b_2\\a_3*b_3\end{pmatrix}" title="\begin{pmatrix}a_0\\a_1\\a_2\\a_3\end{pmatrix} * \begin{pmatrix}b_0\\b_1\\b_2\\b_3\end{pmatrix} \rightarrow \begin{pmatrix}a_0 * b_0\\a_1*b_1\\a_2*b_2\\a_3*b_3\end{pmatrix}" /></a>

Modern AVX transformed the ease of obtaining high performance
* wider registers
* more registers
* many more data types supported
* gather+scatter (operate on non-contiguous data)
* relaxed previous constraints on aligning data in memory
* better support for reductions across a register
* many more operations including math functions (sin, cos, ...)
* fully predicated (historically a key advantage of NVIDIA GPUs over x86 SIMD)

### 6.1 SIMD on a long vector

If the SIMD register is just *W* words wide, how do we operate on a vector of arbitrary length *N*?  The loop is tiled by the compiler
~~~
   for (i=0;i<n;i++) a[i] = b[i]*10.0 + c[i];
~~~
becomes
~~~
   NR = n/W;
   for (R=0; R<NR; R++,i+=W)
      a[i:i+W] = b[i:i+W]*10.0 + c[i:i+W]; // vector op using Python slice notation
   
   for (;i<n;i++) a[i] = b[i]*10.0 + c[i]; // remainder or clean up loop
~~~
In practice, things can be much more complex due to handling address misalignment, unrolling, etc.

### 6.2 Sum example revisited --- looking at the assembly language

Now we understand a bit more, let's look under the hood at what the compiler is doing in the sum example.  

**Exercise:** Look back again at the optimization report.  You should now understand why the projected speed might be estimated as 8 and what it means by remainder loop.

The report said that it vectorized the code, but what did it actually do? Let's look at the assembly code it generated.  

Since the assembly code can be *huge* it helps to use a bit of voodoo to insert comments around the bit we are interested in.  Modify `sum.cc` as follows (insert the two `__asm__` lines)
~~~
#include <iostream>

int main() {
    const int N=100007;
    double a[N];
    __asm__("/*startloop*/");
    for (int i=0; i<N; i++) a[i] = i;
    double sum = 0.0;
    for (int i=0; i<N; i++) sum += a[i];
    __asm__("/*endloop*/");
    std::cout << sum << std::endl;
    return 0;
}
~~~
We will add these flags to the compilation command
* `-S` generate the assembly language output in `<filename>.s`
* `-c` just compile, don't link

So you will run
~~~
icpc -xHOST -O3 -ipo -no-prec-div -fp-model fast=2 -qopt-zmm-usage=high -qopt-report=5 -qopt-report-phase=vec -S -c sum.cc
~~~

Look in `sum.s` and search for `startloop`.
~~~
...
        /*startloop*/
# End ASM                                                       #5.0
# End ASM
                                # LOE rbx r12 r13 r14 r15
..B1.8:                         # Preds ..B1.9
                                # Execution count [1.00e+00]
        vmovdqu   .L_2il0floatpacket.1(%rip), %ymm5             #6.31     <<<<<<<<<<< Sets up for the tiled loop
        xorl      %eax, %eax                                    #6.5
        vmovdqu   .L_2il0floatpacket.2(%rip), %ymm4             #6.31
        vpxord    %zmm3, %zmm3, %zmm3                           #7.16
        vmovaps   %zmm3, %zmm2                                  #7.16
        vmovaps   %zmm2, %zmm1                                  #7.16
        vmovaps   %zmm1, %zmm0                                  #7.16
        .align    16,0x90
                                # LOE rax rbx r12 r13 r14 r15 ymm4 ymm5 zmm0 zmm1 zmm2 zmm3
..B1.2:                         # Preds ..B1.2 ..B1.8 
                                # Execution count [1.00e+02]
        vpaddd    %ymm5, %ymm4, %ymm7                           #6.31     <<<<<<<<<<< Start of the unrolled inner loop
        vcvtdq2pd %ymm4, %zmm6                                  #6.38
        vaddpd    %zmm6, %zmm0, %zmm0                           #8.31
        vpaddd    %ymm5, %ymm7, %ymm9                           #6.31
        vcvtdq2pd %ymm7, %zmm8                                  #6.38
        vaddpd    %zmm8, %zmm3, %zmm3                           #8.31
        vpaddd    %ymm5, %ymm9, %ymm11                          #6.31
        vcvtdq2pd %ymm9, %zmm10                                 #6.38
        vaddpd    %zmm10, %zmm2, %zmm2                          #8.31
        vcvtdq2pd %ymm11, %zmm12                                #6.38
        vpaddd    %ymm5, %ymm11, %ymm4                          #6.31
        vaddpd    %zmm12, %zmm1, %zmm1                          #8.31
        addq      $32, %rax                                     #6.5
        cmpq      $96, %rax                                     #6.5
        jb        ..B1.2        # Prob 99%                      #6.5       <<<<<<<<<<< End of the unrolled inner loop
                                # LOE rax rbx r12 r13 r14 r15 ymm4 ymm5 zmm0 zmm1 zmm2 zmm3
..B1.3:                         # Preds ..B1.2
                                # Execution count [1.00e+00]
        vaddpd    %zmm3, %zmm0, %zmm0                           #7.16
        vaddpd    %zmm1, %zmm2, %zmm1                           #7.16
        vaddpd    %zmm1, %zmm0, %zmm2                           #7.16
        vmovups   %zmm2, (%rsp)                                 #7.16[spill]
                                # LOE rbx r12 r13 r14 r15
..B1.13:                        # Preds ..B1.3
                                # Execution count [1.00e+00]
# Begin ASM
# Begin ASM
        /*endloop*/
...
~~~

It is often hard to understand the assembly code since the compiler knows a lot more about the machine than you.  However, it knows a lot less about your intentions than you and so the code might still be suboptimal. Nevertheless, this is clearly vector code that is somehow mixing use of the 512-bit `zmm*` and 256-bit `ymm*` registers.

### 6.2 Predication

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

|Operation and sources/targets             | Description                                             |
|-----------------------------------------|---------------------------------------------------------|
|VADDPD zmm1 {k1}, zmm2,zmm3/m512/m64bcst |  Add packed double-precision floating-point values from zmm3/m512/m64bcst to zmm2 and store result in zmm1 with writemask k1 |
|VFMADD213PD zmm1 {k1},zmm2,zmm3/m512/m64bcst |  Multiply packed double-precision floating-point values from zmm1 and zmm2, add to zmm3/m512/m64bcst and put result in zmm1 with write mask k1 |


Fortunately, we humans no longer need to write in assembly language.


### 6.3 Memory alignment

Historically, a load into vector register of width *W* bytes could only be performed from memory addresses that were appropriately aligned (usually on an address that was a multiple of *W*).  Subsequently, one could perform unaligned loads but only with a severe performance penalty.  Most recently, unaligned loads suffer a much lower peformance impact, but there can still benefit from aligning data.

For best performance, try to align data structures appropriately and to pad leading dimensions of multi-dimension structures accordingly.  Since this is not a big concern on modern x86 we won't dwell upon it.
* Declare C++ variables using `alignas(W)`
* Allocate on heap with `posix_memalign()` or `std::aligned_alloc()` (C++ 17)
* Multi-dimension array `a[10][7]` perhaps best stored as `a[10][8]` for AVX

Reference
* Intel [Data Alignment to Assist Vectorization](https://software.intel.com/en-us/articles/data-alignment-to-assist-vectorization)



### 6.4 Pipelined SIMD

A factor of *L\*W* speedup compared to serial code.



### 6.5 Exercises

**Exercise:** what is the peak spead (operations/element/cycle) of a single, piplelined, SIMD functional unit with width *W=8* and latency *L=3*? 8.

**Exercise:** how long must your vector be to obtain 90% of peak speed from a single, piplelined, SIMD functional unit with width *W=8* and latency *L=3*?  8*9*(3-1) = 144.   


### 7. Peak speed of Intel Skylake (`sn-mem`)

What is the peak floating performance of a single core of sn-mem, and what do you have to do to realize it?

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

The Wikipedia page on [Skylake](https://en.wikipedia.org/wiki/Skylake_(microarchitecture)) is not especially helpful.  For the full technical details (much more than most of us need) see the links provided above and also
* [Skylake details @ NASA](https://www.nas.nasa.gov/hecc/support/kb/skylake-processors_550.html)


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


**Exercise:** Repeat the analysis for the Seawulf login node


**Exercise:** repeat the analysis for your laptop


## 8.0 Benchmarking DAXPY

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

### 8.1 Observations

In the figure are cycles/element for DAXY as a function of N (the x-axis being on a log scale)

![measured](https://github.com/wadejong/Summer-School-Materials/blob/master/VectorSIMD/plot.gif  "DAXPY cycles/element")

### 8.2 Analysis

Theoretical peak speed on Skylake --- in one cycle can issue 
* FMA AVX512 SIMD
* 2 loads AVX512 SIMD
* 1 store AVX512 SIMD
* loop counter increment
* test and branch

So the entire loop can be written to execute in just one cycle, so the challenge is actually getting data from cache/memory to the processor.  This would be a throughput of 0.125 cyles/element.  CBLAS best is 0.163, likely due to the L1 cache not being able to sustain the peak speed due to bank conflicts.

For small vector lengths we are dominated by overheads (loop, call, timing, ...).  On the figure we see break points at ~2K, ~60K, ~1M.
* L1 cache (private to each core) can, at least in theory, support full-speed memory access.  Also, the L1 cache is 32KB or 2048*16bytes (we have 2 d.p. vectors, so 2028 is the maximum we can fit, but in practice collisions will mean less fits).
* L2 cache (private to each core) delivers in practice at about half the bandwidth of L1.  It is 1MB or 65,536*16bytes.
* L3 cache (shared) runs about 4x slower than L2.  It is about 24MB or 1,622,025*16bytes.

[Aside: nice detailed analysis of Haswell cache access in the comments [here](https://software.intel.com/en-us/forums/intel-moderncode-for-parallel-architectures/topic/608964).]

Here is a plot of the above integrated as a roofline model with the experimental data (converted to being the rate in elements/cycle)

![measured](https://github.com/wadejong/Summer-School-Materials/blob/master/VectorSIMD/roof.gif  "DAXPY elements/cycle")

**Exercise:** instead of timing the `cblas` routine time the compiled code (i.e., comment out `cblas_daxpy` and uncomment the loop)

**Exercise:** in the `bench` directory, read and run the `stride` program that measures the number of cycles/element for this loop
~~~
    for (int i=0; i<n; i+=stride) {
      y[i] += a*x[i];
    }
~~~
that now has non-unit stride (i.e., non-contiguous memory access) and uses a large value of `n`.  Explain what you observe making reference to the size of a cache line (64 bytes) and the page size (4096 bytes).

Here's my results in case you cannot get it running.

![measured](https://github.com/wadejong/Summer-School-Materials/blob/master/VectorSIMD/stride.gif  "DAXPY cycles/element")


## 9.0 Non-trivial example --- vectorizing Metropolis Monte Carlo

Look [here](https://github.com/wadejong/Summer-School-Materials/blob/master/Examples/Vectorization/mc)

## 10.0 Exercises

**Exercise:** Read the Intel vectorization documentation in the links section

**Exercise:** Skim (!) the Intel 64 and IA-32 Architectures Optimization Reference Manual in the links section, just to get a flavor.

**Exercise:** Make sure you understand the Monte Carlo example

**Exercise:** Explain the performance obtained from the matrix multiplication example.  For instance,
~~~
    [rharrison@sn-mem mxm]$ ./mxm
    A(m,k)*B(k,n) --> C(m,n)
    Input m n k: 1024 1024 1024
    basic FLOPS/cycle 5
    daxpy FLOPS/cycle 2.2
     ddot FLOPS/cycle 2.7
      mkl FLOPS/cycle 27
~~~

**Exercise:** Try compiling some of your own code to see how well it vectorizes, and then try to optimize it.



On GPU timings made with optimal parameters
~~~
const int NTHR_PER_BLK = 1024; // Highest number of CUDA threads per block
const int NBLOCK  = 56;  // Number of CUDA blocks (=#SMs on P100)
const int Npoint = NBLOCK*NTHR_PER_BLK; // No. of independent samples
const int Nsample = 100; // No. of blocks to sample
~~~
On CPU timings made with optimal parameters
~~~
const int Npoint = 256;          // No. of independent samples (fits into L1)
const int Nsample = 100*4*56;    // No. of blocks to sample
~~~
* Note that `Npoint*Nsample` is the same on both machines so they are doing the same amount of work.
* Intel timings on sn-mem (Skylake 6154 gold, 3Ghz)
* NVIDIA timings on P100
* Double precision on all platforms (empirically float gives ~2x speedup on both machines)

|P100|Vec|Seq|OMP+vec|
|---|---|---|---|
|13.3s|710s|7100s est.|10s est.|

So, comparing the performance of the different software implementations
on the different hardware, the entire P100 is about 530x faster than
the original unvectorized code on runing on a single core.

Vectorization gives a factor 10 speedup, so the entire P100 is about 53x faster than
the vectorized code on runing on a single core.

We expect near linear speed up on this example using OpenMP
(since the algorithm is massively parallel and all memory references
are localized to core-private caches), so we can estimate that the
entire 72-core machine to be ~1.3x faster than the P100.

This lines up perfectly (!) with the ratio of their peak speeds, which are
~5.3TFLOP/s for the P100 and for 6.9 TFLOP/s for the entire 72-core
Skylake machine.

Conclusion: 
* This benchmark has performance that tracks the peak performance on both machines
* A fair comparison of machines requires careful attention to optimizing the software on each platform, which can take a lot of work.










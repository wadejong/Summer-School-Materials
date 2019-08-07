
Timings made with
~~~
const int NTHR_PER_BLK = 512; // Number of CUDA threads per block
const int NBLOCK  = 56;  // Number of CUDA blocks (SMs on P100)
const int Npoint = NBLOCK*NTHR_PER_BLK; // No. of independent samples
~~~
* Intel timings on sn-mem (Skylake 6154 gold, 3Ghz)
* Double precision on all platforms

|P100|Vec|Seq|
|---|---|---|
|20.7s|394.9s|4000s est.|

So, comparing the performance of the different software implementations
on the different hardware, the entire P100 is 19x faster than a single
Skylake core with vectorized code, and about 190x faster than the
original unvectorized code on a single core.

We expect near linear speed up on this example using OpenMP
(since the algorithm is massively parallel and all memory references
are localized to core-private caches), so we can estimate that the
entire 72-core machine to be ~3.8x faster than the P100.

This seems implausible.  The peak d.p. speed of the P100 is ~5.3TFLOP/s and for the entire 72-core Skylake machine is 6.9 TFLOP/s. So, I would expect them to be running at about the same speed on this benchmark.

**Conclusion::** There is still some optimization to do on the P100 --- about a factor of 3.5x.










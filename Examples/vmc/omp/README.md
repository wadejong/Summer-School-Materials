Now derive the OpenMP parallel version from the sequential version

[An alternative approach is to make the filescope global generator threadprivate but this is encouraging use of global data which is usually not good, so let's pursue a better approach].


First we have to fix some warts in the sequential code

1. Move i loop outside of initialize so it works the same as propagate

2. Move mutable global data back into main program (u, naccept, nreject)

Then,

1. Need a generator for each thread with persisent state initialized to sample an independent stream --- this choice was slow due to overhead of calling omp_get_thread_num() --- use a generator per point.

2. Parallelize the loops setting appropriate type for each variable. 

    * Static (default) schedule was much faster than guided.

3. What might be causes of it running slowly?

4. Using the system command `time` to measure the time is not ideal.  Need to introduce internal measurement of the time.

| N  | realtime | speedup |
|:--:|:--------:|:-------:|
| 1  |  46.2    |  1.0 |
| 2  |  24.8    |  1.86|
| 3  |  18.0    |  2.57|
| 4  |  14.0    |  3.30|
| 5  |  12.4    |  3.75|
| 8  |   9.8    |  4.71|
|10  |   8.6    |  5.37|
|16  |  7.86    |  5.88|


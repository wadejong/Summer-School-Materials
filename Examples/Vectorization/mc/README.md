
# Vectorization of a simple Metropolis Monte Carlo Program

Key ideas:
1.  Loop interchange
1.  Promoting data structures
1.  Strength reduction
1.  Loop spliting
1.  Inlining kernels for readability
1.  Eliminating dependencies
1.  Vectorization of predicates --- how does this work?
1.  Using vector math libraries
1.  Experiment with precision
1.  20x speedup --- ~10x from vectorization, ~2x from precision --- with no loss of accuracy and still using just 1 core!

For this section we will be using the latest Intel Compiler so please execute this command in your shell to load the right modules
~~~
   source /gpfs/projects/molssi/modules-intel
~~~
The latest GNU compiler is actually also pretty good at vectorization, but unfortunately it also needs an up-to-date GLIBC and the one currently installed on Seawulf is too old.

The [Metropolis Monte Carlo (MMC)](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) is a powerful algorithm to compute complex, high-dimensional integrals.  It's used extensively in both computational science and data analytics including Bayesian inference.  Briefly, given a computable function <a href="https://www.codecogs.com/eqnedit.php?latex=p(x)>=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)>=0" title="p(x)>=0" /></a> we can interpet <a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;/&space;\int&space;p(x)&space;dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)&space;/&space;\int&space;p(x)&space;dx" title="p(x) / \int p(x) dx" /></a> as a normalized probability distribution.  MMC is a recipe for (asymptotically) sampling points <a href="https://www.codecogs.com/eqnedit.php?latex=x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i" title="x_i" /></a> from this distribution so that you can estimate integrals via the expectation value

<a href="https://www.codecogs.com/eqnedit.php?latex=<&space;f&space;>&space;=&space;\int&space;f(x)&space;p(x)&space;dx&space;=&space;\frac{1}{N}\sum_{i=0}^{N-1}&space;f(x_i)&space;&plus;&space;O(N^{-1/2})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?<&space;f&space;>&space;=&space;\int&space;f(x)&space;p(x)&space;dx&space;=&space;\frac{1}{N}\sum_{i=0}^{N-1}&space;f(x_i)&space;&plus;&space;O(N^{-1/2})" title="< f > = \int f(x) p(x) dx = \frac{1}{N}\sum_{i=0}^{N-1} f(x_i) + O(N^{-1/2})" /></a>

The algorithm is very easy.  Start with by sampling a random point <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a> distributed in the computational volume.
1. Sample a new random point <a href="https://www.codecogs.com/eqnedit.php?latex=x^\prime" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^\prime" title="x^\prime" /></a>
1. With probability <a href="https://www.codecogs.com/eqnedit.php?latex=P&space;=&space;\min(1,p(x^\prime)/p(x))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P&space;=&space;\min(1,p(x^\prime)/p(x))" title="P = \min(1,p(x^\prime)/p(x))" /></a> accept the new value (i.e., <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;x^\prime" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;x^\prime" title="x = x^\prime" /></a>)
1. Repeat steps 1 and 2 until the asymptotic distribution is reached --- at this time you can start computing the desired statistics.

A key refinement is to compute with multiple independent samples of <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;x^\prime" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;x^\prime" title="x = x^\prime" /></a> in order to reduce correlation betwen samples.

We will abuse this powerful method to evaluate in 1D

<a href="https://www.codecogs.com/eqnedit.php?latex=\int_0^\infty&space;x&space;e^{-x}&space;dx&space;=&space;<x>&space;\&space;\&space;\mbox{with}\&space;x\&space;\mbox{sampled&space;from}\&space;e^{-x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\int_0^\infty&space;x&space;e^{-x}&space;dx&space;=&space;<x>&space;\&space;\&space;\mbox{with}\&space;x\&space;\mbox{sampled&space;from}\&space;e^{-x}" title="\int_0^\infty x e^{-x} dx = <x> \ \ \mbox{with}\ x\ \mbox{sampled from}\ e^{-x}" /></a>

by sampling points from <a href="https://www.codecogs.com/eqnedit.php?latex=e^{-x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e^{-x}" title="e^{-x}" /></a> and computing the expectation value of <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a> (for which we know the value is exactly one).

The core part of the sampling algorithm to sample is just

~~~
   1.  x = 23.0*rand() // initialize
   2.  while (1) {
   3.     xnew = 23.0*rand();
   4.     if (exp(-xnew) > exp(-x)*rand()) x = xnew;
   5.     sum += x
   6.  }
~~~
Instead of computing over a semi-infinite domain we compute over *[0,23]* since *exp(-23)~1e-10* which is negligble for present purposes. Also, we assume `rand()` is a function that returns a number uniformly distributed over *[0,1]*.
1. Samples the initial point
2. Starts the infinite loop
3. Samples the new point
4. Perfoms the acceptance test
5. Accumulates the statistics (need a "warmup period" before doing this)

**Questions so far?**

Now we go to the source directory in [`Vectorization/mc`](https://github.com/wadejong/Summer-School-Materials/blob/master/Examples/Vectorization/mc) and look in turn at the files `mc[0-5].cc`.

**Exercise (hard)**: Vectorize the variational quantum Monte Carlo program in [`Examples/vmc`](https://github.com/wadejong/Summer-School-Materials/blob/master/Examples/vmc).













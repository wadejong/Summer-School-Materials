
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
1.  20x speedup --- 10x from vectorization, 2x from precision --- with no loss of accuracy

The Metropolis Monte Carlo (MMC) is a powerful algorithm to compute complex, high-dimensional integrals.  It's used extensively in both computational science and data analytics including Bayesian inference.  Briefly, given a computable function <a href="https://www.codecogs.com/eqnedit.php?latex=p(x)>=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)>=0" title="p(x)>=0" /></a> we can interpet <a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;/&space;\int&space;p(x)&space;dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)&space;/&space;\int&space;p(x)&space;dx" title="p(x) / \int p(x) dx" /></a> as a normalized probability distribution.  MMC is a recipe for (asymptotically) sampling points <a href="https://www.codecogs.com/eqnedit.php?latex=x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i" title="x_i" /></a> from this distribution so that you can estimate integrals via the expectation value

<a href="https://www.codecogs.com/eqnedit.php?latex=\int&space;f(x)&space;p(x)&space;dx&space;\approx&space;\sum_{i=0}^{N-1}&space;f(x_i)&space;&plus;&space;O(N^{-1/2})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\int&space;f(x)&space;p(x)&space;dx&space;\approx&space;\sum_{i=0}^{N-1}&space;f(x_i)&space;&plus;&space;O(N^{-1/2})" title="\int f(x) p(x) dx \approx \sum_{i=0}^{N-1} f(x_i) + O(N^{-1/2})" /></a>

The algorithm is very easy.  Start with by sampling a random point <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a> distributed uniformly in the computational volume.
1. Sample a new random point <a href="https://www.codecogs.com/eqnedit.php?latex=x^\prime" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^\prime" title="x^\prime" /></a>
1. With probability <a href="https://www.codecogs.com/eqnedit.php?latex=P&space;=&space;\min(1,p(x^\prime)/p(x))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P&space;=&space;\min(1,p(x^\prime)/p(x))" title="P = \min(1,p(x^\prime)/p(x))" /></a> accept the new value (i.e., <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;x^\prime" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;x^\prime" title="x = x^\prime" /></a>)
1. Repeat steps 1 and 2 until the asymptotic distribution is reached --- at this time you can start computing the desired statistics.

A key refinement is to compute with multiple independent samples of <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;x^\prime" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;x^\prime" title="x = x^\prime" /></a> in order to reduce correlation betwen samples.

We will abuse this powerful method to evaluate in 1D

<a href="https://www.codecogs.com/eqnedit.php?latex=\int_0^\infty&space;x\exp(-x)&space;dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\int_0^\infty&space;x&space;exp(-x)&space;dx" title="\int_0^\infty x exp(-x) dx" /></a>

for which we know the value is exactly one.

The key part of the algorithm to sample is just

~~~
   x = 23.0*rand() // initialize
   while (1) {
      xnew = 32.0*rand();
      if (exp(-xnew) > exp(-x)*rand()) x = xnew;
   }
~~~











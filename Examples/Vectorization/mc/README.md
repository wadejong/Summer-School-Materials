
# Vectorization of a simple Metropolis Monte Carlo Program

Key ideas:
1.  Loop interchange
1.  Promoting data structures
1.  Loop spliting
1.  Inlining kernels for readability
1.  Eliminating dependencies
1.  Using vector math libraries
1.  Experiment with precision
1.  20x speedup --- 10x from vectorization, 2x from precision --- with no loss of accuracy

<a href="https://www.codecogs.com/eqnedit.php?latex=\int_0^\infty&space;exp(-x)&space;dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\int_0^\infty&space;exp(-x)&space;dx" title="\int_0^\infty exp(-x) dx" /></a>





This program computes expectation values (6D integrals) associated with the helium atom.  

Define *r1*, *r2*, and *r12* to be the distances of the electrons from the nucleus and from each other. 

The classic Hylleraas wavefunction for the helium atom is

<a href="https://www.codecogs.com/eqnedit.php?latex=\psi(r_1,r_2,r_{12})&space;=&space;(1&space;&plus;&space;\frac{1}{2}&space;r_{12})&space;\exp(-2(r_1&plus;r_2))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\psi(r_1,r_2,r_{12})&space;=&space;(1&space;&plus;&space;\frac{1}{2}&space;r_{12})&space;\exp(-2(r_1&plus;r_2))" title="\psi(r_1,r_2,r_{12}) = (1 + \frac{1}{2} r_{12}) \exp(-2(r_1+r_2))" /></a>

We want to compute *\<r1\>*, *\<r2\>*, *\<r12\>* where

<a href="https://www.codecogs.com/eqnedit.php?latex=<f>&space;=&space;\frac{1}{N}\int&space;f(r_1,r_2,r_{12})\psi(r_1,r_2,r_{12})^2&space;dx_1&space;dy_1&space;dz_1&space;dx_2&space;dy_2&space;dz_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?<f>&space;=&space;\frac{1}{N}\int&space;f(r_1,r_2,r_{12})\psi(r_1,r_2,r_{12})^2&space;dx_1&space;dy_1&space;dz_1&space;dx_2&space;dy_2&space;dz_2" title="<f> = \frac{1}{N}\int f(r_1,r_2,r_{12})\psi(r_1,r_2,r_{12})^2 dx_1 dy_1 dz_1 dx_2 dy_2 dz_2" /></a>

where *N* is the constant that square normalizes *psi*.

This is performed using the Metropolis algorithm that interprets <a href="https://www.codecogs.com/eqnedit.php?latex=p&space;=&space;\psi^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p&space;=&space;\psi^2" title="p = \psi^2" /></a> as an unormalized probability distribution.  Points are asymptotically sampled from the distribution p() as follows

1. initialize vectors *r1* and *r2* and compute *p=psi^2*, then repeat steps 2 and 3

2. randomly sample new vectors *r1*, *r2* and compute new value of *p*

3. accept the new point with probablilty *min(1,p/pnew)*

Once the population has converged (in this simple example we assume this happens after Neq steps) you can start computing statistics.


The sub-directories contain the following:

* seq --- the original sequential (unvectorized) version

* omp --- the open mp version

* mpi --- the mpi version

* vec --- the vectorized version

* cuda --- the CUDA version for NVIDIA GPUs


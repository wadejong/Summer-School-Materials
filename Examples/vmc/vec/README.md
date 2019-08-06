# Vectorized VMC example code

This version vectorizes the code in `Examples/vmc/seq` using nothing more than the techniques discussed in the 1-D Monte Calor example in `Examples/Vectorization/mc`.  It's a little messier just because we are now computing in 6D (two electrons in 3D space) so instead of having one coordinate *x* (as it was called in the 1D MC example) we have a vector of 6 numbers `r[6]`.

The `vmc.cc` sequential code already had an inside loop over independent points with an array holding the position of each walkeer that was essentially 'R[Npoint][6]' (in C/C++ ordering in which the last dimension is stored contiguously, which is the opposite to Fortran).  However, to see the problem with this imagine operating on the *x1* coordinate of point *i* that would be stored in 'R[i][0]' --- a vectorized loop operating on the *x1* coordinate needs the value of *x1* for *all* points to be contiguous in memory.  However, in the original storage the actual stride between values of *x1* is 6, not 1.  

So we need to change the layout.  What I probably should have done is to just switch to use `R[6][Npoint]` but instead I created separate arrays for `x1[Npoint]`, `y[Npoint]`, etc.

Other than this, if you understand what happened in the 1D MC example, this should be straightforward.

The performance gain is pretty amazing --- **10x** ! Clearly worth the effort.




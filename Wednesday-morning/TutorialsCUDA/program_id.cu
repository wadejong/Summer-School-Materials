#include <stdio.h>
#include <cuda.h>

//-----------------------------------------------------------------------------
// TheKernel: basic kernel containing a print statement.
//-----------------------------------------------------------------------------
__global__ void TheKernel()
{
  // The variable "threadIdx" is given to the programmer as part of the
  // CUDA environment.  It is a C-style struct, with three "dimensions."
  // (The thread block can have up to three axes.) Most scientific
  // applications need only the first ('x'), because it's not necessary to
  // organize threads into two-dimensional bundles.  For image processing,
  // two dimensions are nice to have.
  printf("This is kernel thread %2d saying hello world, from the GPU.\n",
	 threadIdx.x);
}

//-----------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------
int main()
{
  printf("This is the C layer saying hello world, from the host.\n");

  // Now add more than just one thread to the kernel
  TheKernel<<<1, 32>>>();

  // Device synchronization
  cudaDeviceSynchronize();

  // Announce that the kernel is complete
  printf("Program exits.\n");
  
  return 0;
}

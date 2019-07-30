#include <stdio.h>
#include <cuda.h>

// Notice that this file needs to have a .cu extension for the NVCC compiler
// to understand what it is supposed to do.  NVCC can compile C and C++, by
// emulating a C++ compiler.  However, any code that contains GPU kernels
// must reside in a CUDA unit with .cu extension.

//-----------------------------------------------------------------------------
// TheKernel: basic kernel containing a print statement.
//-----------------------------------------------------------------------------
__global__ void TheKernel()
{
  printf("This is the kernel saying hello world, from the GPU.\n");
}

//-----------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------
int main()
{
  printf("This is the C layer saying hello world, from the host.\n");

  // Launch the kernel
  TheKernel<<<1, 1>>>();

  // It appears essential to call for synchronization before finally
  // exiting, lest you risk the program crashing your machine!
  cudaDeviceSynchronize();

  return 0;
}

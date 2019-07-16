#include <stdio.h>
#include <cuda.h>

//-----------------------------------------------------------------------------
// TheKernel: basic kernel containing a print statement.
//-----------------------------------------------------------------------------
__global__ void TheKernel()
{
  // Reduce the output: x & 7 means "compare the 32-bit string x (an integer)
  // to 0x7 (hexadecimal for 7), which is 0000 0000 ... 0000 0111, returning
  // 1 if both bits read "one" and 0 if either bit reads "zero." This masks
  // all but the final three bits of x, which is the same as taking mod(x, 8).
  // In general, (x & (2^N - 1)) = mod(x, 2^N).
  if ((threadIdx.x & 7) == 0) {
    printf("This is block %2d, thread %2d saying hello world, from the GPU.\n",
           blockIdx.x, threadIdx.x);
  }
}

//-----------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------
int main()
{
  printf("This is the C layer saying hello world, from the host.\n");

  // Launch the kernel in more than one block
  TheKernel<<<8, 32>>>();

  // Device synchronization
  cudaDeviceSynchronize();

  // Announce that the kernel is complete
  printf("Program exits.\n");
  
  return 0;
}

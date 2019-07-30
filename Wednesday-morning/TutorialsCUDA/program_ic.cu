#include <stdio.h>
#include <cuda.h>

//-----------------------------------------------------------------------------
// TheKernel: basic kernel containing a print statement.
//-----------------------------------------------------------------------------
__global__ void TheKernel()
{
  // Give the kernel something to keep its (single) thread occupied
  int i, j, k;
  k = 0;
  for (i = 0; i < 1000; i++) {
    for (j = 0; j < 1000; j++) {
      k += i;
      if (k > 2000) {
	k -= 4*j;
      }
      else {
	k += j;
      }
    }
  }
  
  printf("This is the kernel saying hello world, from the GPU.\n");
}

//-----------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------
int main()
{
  printf("This is the C layer saying hello world, from the host.\n");
  TheKernel<<<1, 1>>>();

  // Device synchronization
  cudaDeviceSynchronize();
  printf("LOOK: device synchronization stops the host until the kernel is "
	 "done.\n");
  
  // It appears essential to call for synchronization before finally
  // exiting, lest you risk the program crashing your machine!
  cudaDeviceSynchronize();

  return 0;
}

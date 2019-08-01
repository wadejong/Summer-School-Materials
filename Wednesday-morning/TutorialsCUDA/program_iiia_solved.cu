#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

//-----------------------------------------------------------------------------
// GpuConstantsPackage: a struct to hold many constants (including pointers
//                      to allocated memory on the device) that can be
//                      uploaded all at once.  Placing this in the "constants
//                      cache" is a convenient and performant way of handling
//                      constant information on the GPU.
//-----------------------------------------------------------------------------
struct GpuConstantsPackage {
  int     nvalue;
  int*    values;
};
typedef struct GpuConstantsPackage dataPack;

// This device constant is available to all functions in this CUDA unit
__device__ __constant__ dataPack dPk;

//-----------------------------------------------------------------------------
// GpuMirroredInt: a struct holding mirrored int data on both the CPU and the
//                 GPU.  Functions below will operate on this struct
//                 (because this isn't a workshop on C++)
//-----------------------------------------------------------------------------
struct GpuMirroredInt {
  int len;          // Length of the array (again, this is not a C++ course)
  int IsPinned;     // "Pinned" memory is best for Host <= => GPU transfers.
                    //   In fact, if non-pinned memory is transferred to the
                    //   GPU from the host, a temporary allocation of pinned
                    //   memory will be created and then destroyed.  Pinned
                    //   memory is not host-pageable, but the only performance
                    //   implication is that creating lots of pinned memory
                    //   may make it harder for the host OS to manage large
                    //   memory jobs.
  int* HostData;    //   Pointer to allocated memory on the host
  int* DevcData;    //   Pointer to allocated memory on the GPU.  Note that the
                    //     host can know what the address of memory on the GPU
                    //     is, but it cannot simply de-reference that pointer
                    //     in host code.
};
typedef struct GpuMirroredInt gpuInt;

//-----------------------------------------------------------------------------
// GpuMirroredInt: a struct holding mirrored fp32 data on both the CPU and the
//                 GPU.  Functions below will operate on this struct
//                 (because this isn't a workshop on C++)
//-----------------------------------------------------------------------------
struct GpuMirroredFloat {
  int len;          // Length of the array (again, this is not a C++ course)
  int IsPinned;     // "Pinned" memory is best for Host <= => GPU transfers.
                    //   In fact, if non-pinned memory is transferred to the
                    //   GPU from the host, a temporary allocation of pinned
                    //   memory will be created and then destroyed.  Pinned
                    //   memory is not host-pageable, but the only performance
                    //   implication is that creating lots of pinned memory
                    //   may make it harder for the host OS to manage large
                    //   memory jobs.
  float* HostData;  //   Pointer to allocated memory on the host
  float* DevcData;  //   Pointer to allocated memory on the GPU.  Note that the
                    //     host can know what the address of memory on the GPU
                    //     is, but it cannot simply de-reference that pointer
                    //     in host code.
};
typedef struct GpuMirroredFloat gpuFloat;

//-----------------------------------------------------------------------------
// kWarpPrefixSum: kernel for making a prefix sum of 32 numbers
//-----------------------------------------------------------------------------
__global__ void kWarpPrefixSum()
{
  if (threadIdx.x == 0) {
    printf("Values =\n");
    int i, j;
    for (i = 0; i < 4; i++) {
      printf("    ");
      for (j = 8*i; j < 8*(i+1); j++) {
        printf("%4d ", dPk.values[j]);
      }
      printf(" [ slots %2d - %2d ]\n", 8*i, 8*(i+1)-1);
    }
  }

  int tgx = (threadIdx.x & 31);
  int var = dPk.values[threadIdx.x];
  var += ((tgx &  1) ==  1) * __shfl_up_sync(0xffffffff, var, 1);
  var += ((tgx &  3) ==  3) * __shfl_up_sync(0xffffffff, var, 2);
  var += ((tgx &  7) ==  7) * __shfl_up_sync(0xffffffff, var, 4);
  var += ((tgx & 15) == 15) * __shfl_up_sync(0xffffffff, var, 8);
  var += (tgx == 31) * __shfl_up_sync(0xffffffff, var, 16);
  var += ((tgx & 15) == 7 && tgx > 16) * __shfl_up_sync(0xffffffff, var, 8);
  var += ((tgx &  7) == 3 && tgx > 8)  * __shfl_up_sync(0xffffffff, var, 4);
  var += ((tgx &  3) == 1 && tgx > 4)  * __shfl_up_sync(0xffffffff, var, 2);
  var += ((tgx &  1) == 0 && tgx >= 2) * __shfl_up_sync(0xffffffff, var, 1);
  dPk.values[threadIdx.x] = var;
  __syncthreads();

  if (threadIdx.x == 0) {
    printf("Values =\n");
    int i, j;
    for (i = 0; i < 4; i++) {
      printf("    ");
      for (j = 8*i; j < 8*(i+1); j++) {
	printf("%4d ", dPk.values[j]);
      }
      printf(" [ slots %2d - %2d ]\n", 8*i, 8*(i+1)-1);
    }
  }  
}

//-----------------------------------------------------------------------------
// CreateGpuInt: constructor function for allocating memory in a gpuInt
//               instance.
//
// Arguments:
//   len:      the length of array to allocate
//   pin:      flag to have the memory pinned (non-pageable on the host side
//             for optimal transfer speed ot the device)
//-----------------------------------------------------------------------------
gpuInt CreateGpuInt(int len, int pin)
{
  gpuInt G;

  G.len = len;
  G.IsPinned = pin;
  
  // Now that the official length is recorded, upgrade the real length
  // to the next convenient multiple of 128, so as to always allocate
  // GPU memory in 512-byte blocks.  This is for alignment purposes,
  // and keeping host to device transfers in line.
  len = ((len + 127) / 128) * 128;
  if (pin == 1) {
    cudaHostAlloc((void **)&G.HostData, len * sizeof(int),
		  cudaHostAllocMapped);
  }
  else {
    G.HostData = (int*)malloc(len * sizeof(int));
  }
  cudaMalloc((void **)&G.DevcData, len * sizeof(int));
  memset(G.HostData, 0, len * sizeof(int));
  cudaMemset((void *)G.DevcData, 0, len * sizeof(int));

  return G;
}

//-----------------------------------------------------------------------------
// DestroyGpuInt: destructor function for freeing memory in a gpuInt
//                instance.
//-----------------------------------------------------------------------------
void DestroyGpuInt(gpuInt *G)
{
  if (G->IsPinned == 1) {
    cudaFreeHost(G->HostData);
  }
  else {
    free(G->HostData);
  }
  cudaFree(G->DevcData);
}

//-----------------------------------------------------------------------------
// UploadGpuInt: upload an integer array from the host to the device.
//-----------------------------------------------------------------------------
void UploadGpuInt(gpuInt *G)
{
  cudaMemcpy(G->DevcData, G->HostData, G->len * sizeof(int),
             cudaMemcpyHostToDevice);
}

//-----------------------------------------------------------------------------
// DownloadGpuInt: download an integer array from the host to the device.
//-----------------------------------------------------------------------------
void DownloadGpuInt(gpuInt *G)
{
  cudaMemcpy(G->HostData, G->DevcData, G->len * sizeof(int),
	     cudaMemcpyHostToDevice);
}

//-----------------------------------------------------------------------------
// CreateGpuFloat: constructor function for allocating memory in a gpuFloat
//                 instance.
//
// Arguments:
//   len:      the length of array to allocate
//   pin:      flag to have the memory pinned (non-pageable on the host side
//             for optimal transfer speed ot the device)
//-----------------------------------------------------------------------------
gpuFloat CreateGpuFloat(int len, int pin)
{
  gpuFloat G;

  G.len = len;
  G.IsPinned = pin;
  
  // Now that the official length is recorded, upgrade the real length
  // to the next convenient multiple of 128, so as to always allocate
  // GPU memory in 512-byte blocks.  This is for alignment purposes,
  // and keeping host to device transfers in line.
  len = ((len + 127) / 128) * 128;
  if (pin == 1) {
    cudaHostAlloc((void **)&G.HostData, len * sizeof(float),
		  cudaHostAllocMapped);
  }
  else {
    G.HostData = (float*)malloc(len * sizeof(float));
  }
  cudaMalloc((void **)&G.DevcData, len * sizeof(float));
  memset(G.HostData, 0, len * sizeof(float));
  cudaMemset((void *)G.DevcData, 0, len * sizeof(float));

  return G;
}

//-----------------------------------------------------------------------------
// DestroyGpuFloat: destructor function for freeing memory in a gpuFloat
//                  instance.
//-----------------------------------------------------------------------------
void DestroyGpuFloat(gpuFloat *G)
{
  if (G->IsPinned == 1) {
    cudaFreeHost(G->HostData);
  }
  else {
    free(G->HostData);
  }
  cudaFree(G->DevcData);
}

//-----------------------------------------------------------------------------
// UploadGpuFloat: upload an float array from the host to the device.
//-----------------------------------------------------------------------------
void UploadGpuFloat(gpuFloat *G)
{
  cudaMemcpy(G->DevcData, G->HostData, G->len * sizeof(float),
             cudaMemcpyHostToDevice);
}

//-----------------------------------------------------------------------------
// DownloadGpuFloat: download an float array from the host to the device.
//-----------------------------------------------------------------------------
void DownloadGpuFloat(gpuFloat *G)
{
  cudaMemcpy(G->HostData, G->DevcData, G->len * sizeof(float),
	     cudaMemcpyHostToDevice);
}

//-----------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------
int main()
{
  int i, np;
  gpuInt ivals;
  
  // Create a small array of integers and populate it
  ivals = CreateGpuInt(32, 1);

  // Initialize random number generator
  srand(29538);
  
  // Create random numbers
  np = 32;
  for (i = 0; i < np; i++) {
    ivals.HostData[i] = (int)(100 * (double)rand() / (double)RAND_MAX);
  }

  // Stage critical constants--see cribSheet struct instance cSh above.
  dataPack dpstage;
  dpstage.nvalue = np;
  dpstage.values = ivals.DevcData;

  // Upload all data to the device
  UploadGpuInt(&ivals);

  // Upload the constants to the constants cache
  cudaMemcpyToSymbol(dPk, &dpstage, sizeof(dataPack));  
  
  // Launch the kernel in more than one block
  kWarpPrefixSum<<<1, 32>>>();

  // Device synchronization
  cudaDeviceSynchronize();
  
  return 0;
}

#include <stdio.h>
#include <cuda.h>

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
  int* HostData;    // Pointer to allocated memory on the host
  int* DevcData;    // Pointer to allocated memory on the GPU.  Note that the
                    //   host can know what the address of memory on the GPU
                    //   is, but it cannot simply de-reference that pointer
                    //   in host code.
};
typedef struct GpuMirroredInt gpuInt;

//-----------------------------------------------------------------------------
// WorkbenchKernel: accept data from the host and perform operations on it
//-----------------------------------------------------------------------------
__global__ void WorkbenchKernel(int* devInts)
{
  int i;
  
  // Print from the GPU to show that it sees a copy of the data uploaded
  // from the host.  Note that warp synchronicity is no longer guaranteed
  // in CUDA 9.0 and later versions (we're now on 10.1), which creates the
  // need for all the __syncwarp() calls.
  if (threadIdx.x == 0) {
    printf("The device sees = [\n");
  }
  __syncwarp();
  for (i = 0; i < 16; i++) {
    if (threadIdx.x == i) {
      printf(" %3d", devInts[threadIdx.x]);
    }

    // Synchronization calls (__syncwarp across 32 threads in a warp or
    // __syncthreads across all threads in the block--this block just
    // happens to have only 32 threads) by definition require that all
    // affected threads reach them.  Therefore, they cannot be called
    // from within conditional statements that exclude any relevant
    // threads, i.e. threadIdx.x == some number, seen above.
    __syncwarp();
  }
  if (threadIdx.x == 0) {
    printf(" [index  0-15]\n");
  }
  __syncwarp();
  for (i = 16; i < 32; i++) {
    if (threadIdx.x == i) {
      printf(" %3d", devInts[threadIdx.x]);
    }
    __syncwarp();
  }
  if (threadIdx.x == 0) {
    printf(" [index 16-31]\n];\n");
  }
  __syncthreads();

  // Do some work on the data that can then be inspected on the CPU.
  // Remember, (x & (2^N - 1)) is mod(x, 2^N).
  devInts[threadIdx.x] += (threadIdx.x & 7) * threadIdx.x;
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
// main
//-----------------------------------------------------------------------------
int main()
{
  int i, j;
  gpuInt myInts;
  
  // Create a small array of integers and populate it
  myInts = CreateGpuInt(32, 1);
  for (i = 0; i < 32; i++) {

    // Logical operations such as ((i & 3) == 0) will evaluate to zero or
    // one if they are true or false, and then feed into the arithmetic.
    myInts.HostData[i] = ((i & 3) == 0)*(32 - i) + ((i & 3) != 0)*i;
  }

  // Print the data as originally laid out on the host
  printf("Host data starts as = [\n");
  j = 0;
  for (i = 0; i < 32; i++) {
    printf(" %3d", myInts.HostData[i]);
    j++;
    if (j == 16) {
      printf(" [index  0-15]\n");
    }
  }
  printf(" [index 16-31]\n];\n");

  // Upload data to the device
  UploadGpuInt(&myInts);
  
  // Launch the kernel in more than one block
  WorkbenchKernel<<<1, 32>>>(myInts.DevcData);

  // Download data back to the host
  DownloadGpuInt(&myInts);

  // Print the data as the host now sees it, following work on the GPU 
  printf("Host data now reads = [\n");
  j = 0;
  for (i = 0; i < 32; i++) {
    printf(" %3d", myInts.HostData[i]);
    j++;
    if (j == 16) {
      printf(" [index  0-15]\n");
    }
  }
  printf(" [index 16-31]\n];\n");

  // Device synchronization
  cudaDeviceSynchronize();
  
  return 0;
}

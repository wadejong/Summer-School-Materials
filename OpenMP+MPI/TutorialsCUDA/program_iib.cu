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
  int     nparticle;
  int*    partType;
  float*  partX;
  float*  partY;
  float*  partZ;
  float*  partQ;
  float*  Etot;
};
typedef struct GpuConstantsPackage cribSheet;

// This device constant is available to all functions in this CUDA unit
__device__ __constant__ cribSheet cSh;

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
  float* HostData;  // Pointer to allocated memory on the host
  float* DevcData;  // Pointer to allocated memory on the GPU.  Note that the
                    //   host can know what the address of memory on the GPU
                    //   is, but it cannot simply de-reference that pointer
                    //   in host code.
};
typedef struct GpuMirroredFloat gpuFloat;

//-----------------------------------------------------------------------------
// ParticleSimulator: run a rudimentary simulation of particles
//-----------------------------------------------------------------------------
__global__ void ParticleSimulator()
{
  // Show that the device understands the number of particles
  if (threadIdx.x == 0) {
    printf("There are %d particles active.\n", cSh.nparticle);
  }

  // Loop over all atoms to report positions.  This will NOT necessarily
  // report all particles in orer 0, 1, 2, ..., N.  The second warp
  // (threadIdx.x >= 32 and threadIdx.x < 64) may fire off before the first,
  // printing particles 32, 33, ... before 0, 1, 2, and so forth.  Race
  // conditions are mitigable with good programming practices to insulate
  // code against thread access before results are ready.
  int tidx = threadIdx.x;
  while (tidx < cSh.nparticle) {
    printf("Particle %3d:  %9.4f %9.4f %9.4f  with %9.4f charge\n", tidx,
	   cSh.partX[tidx], cSh.partY[tidx], cSh.partZ[tidx], cSh.partQ[tidx]);
    tidx += blockDim.x;
  }

  // Loop over all particles and compute the electrostatic potential.
  // Each thread will accumulate its own portion of the potential,
  // then pool the results at the end.
  tidx = threadIdx.x;
  float qq = 0.0;
  while (tidx < cSh.nparticle) {

    // Naive way: each thread takes a particle and then loops over all
    // other particles, up to but not including itself.  Low-numbered
    // threads will have little work to do while high-numbered threads
    // will cause the rest of the thread block to idle.
    int i;
    for (i = 0; i < tidx; i++) {
      float dx = cSh.partX[tidx] - cSh.partX[i];
      float dy = cSh.partY[tidx] - cSh.partY[i];
      float dz = cSh.partZ[tidx] - cSh.partZ[i];
      float r = sqrt(dx*dx + dy*dy + dz*dz);
      qq += cSh.partQ[tidx] * cSh.partQ[i] / r;
    }

    // Advance the counter by the number of threads in the block.
    // The kernel of 64 threads processes particles 0-63, 64-127,
    // 128-191, etc. until reaching nparticle.
    tidx += blockDim.x;
  }

  // Each thread's qq could contain information pertaining to multiple atoms
  // each interacting with their respective pairs.  That's fine--the goal is
  // to compute a total potential energy, not anything for one atom in
  // particular.
  atomicAdd(&cSh.Etot[0], qq);

  // For edification, let's try that without atomic instructions: every
  // thread simply accumulates to a number without regard to whether
  // other threads are trying to edit the same number.
  cSh.Etot[1] += qq;
}

//-----------------------------------------------------------------------------
// CreateGpuInt: constructor function for allocating memory in a gpuInt
//               instance.
//
// Arguments:
//   len:      the length of array to allocate
//   pin:      flag to have the memory pinned (non-pageable on the host side
//             for optimal transfer speed to the device)
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
  gpuInt particleTypes;
  gpuFloat particleXcoord, particleYcoord, particleZcoord, particleCharge;
  gpuFloat etot;
  
  // Create a small array of particles and populate it
  particleTypes  = CreateGpuInt(1000, 1);
  particleXcoord = CreateGpuFloat(1000, 1);
  particleYcoord = CreateGpuFloat(1000, 1);
  particleZcoord = CreateGpuFloat(1000, 1);
  particleCharge = CreateGpuFloat(1000, 1);

  // Allocate and initialize the total energy
  // accumulator on the host and on the device.
  etot = CreateGpuFloat(2, 1);
  etot.HostData[0] = 0.0;
  etot.HostData[1] = 0.0;
  UploadGpuFloat(&etot);
  
  // Initialize random number generator.  srand() SEEDS the generator,
  // thereafter each call to rand() will return a different number.
  // This is a reeally bad generator (much better methods with longer
  // periods before they start looping back over the same sequence are
  // available).
  srand(62052);
  
  // We have allocated for a maximum of 100 particles,
  // but let's say there are only 47 in this case.
  np = 47;
  for (i = 0; i < np; i++) {

    // Integer truncation would happen anyway, I'm just making it explicit
    particleTypes.HostData[i] = (int)(8 * rand());

    // Create some random coordinates (double-to-float conversion
    // is happening here.  On the GPU this can have performance
    // impact, so keep an eye on the data types at all times!
    particleXcoord.HostData[i] = 20.0 * (double)rand() / (double)RAND_MAX;
    particleYcoord.HostData[i] = 20.0 * (double)rand() / (double)RAND_MAX;
    particleZcoord.HostData[i] = 20.0 * (double)rand() / (double)RAND_MAX;
    particleCharge.HostData[i] = 0.5 - rand() / (double)RAND_MAX;
  }

  // Stage critical constants--see cribSheet struct instance cSh above.
  cribSheet cnstage;
  cnstage.nparticle = np;
  cnstage.partX = particleXcoord.DevcData;
  cnstage.partY = particleYcoord.DevcData;
  cnstage.partZ = particleZcoord.DevcData;
  cnstage.partQ = particleCharge.DevcData;
  cnstage.Etot  = etot.DevcData; 
  
  // Upload all data to the device
  UploadGpuInt(&particleTypes);
  UploadGpuFloat(&particleXcoord);
  UploadGpuFloat(&particleYcoord);
  UploadGpuFloat(&particleZcoord);
  UploadGpuFloat(&particleCharge);

  // Upload the constants to the constants cache
  cudaMemcpyToSymbol(cSh, &cnstage, sizeof(cribSheet));  
  
  // Launch the kernel
  ParticleSimulator<<<1, 64>>>();

  // Download the total energy
  DownloadGpuFloat(&etot);
  printf("Total energy (atomic adds)   = %10.4f\n", etot.HostData[0]);
  printf("Total energy (straight adds) = %10.4f\n", etot.HostData[1]);
  
  // Device synchronization
  cudaDeviceSynchronize();
  
  return 0;
}

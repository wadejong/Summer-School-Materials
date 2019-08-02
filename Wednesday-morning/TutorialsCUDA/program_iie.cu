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
  float*  partFrcX;
  float*  partFrcY;
  float*  partFrcZ;
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
  int i;

  // Each thread must have __shared__ memory, visible by other threads,
  // to store information about one particle it has been assigned to read
  // and manage.  Each array has as many elements as there are threads in
  // the block.  If the launch parameters were to change, all of these
  // array sizes should change as well.
  __shared__ volatile float   pX[512],   pY[512],   pZ[512], pQ[512];
  __shared__ volatile float   tX[512],   tY[512],   tZ[512], tQ[512];
  __shared__ volatile float sfpX[512], sfpY[512], sfpZ[512];
  __shared__ volatile float sftX[512], sftY[512], sftZ[512];
  
  // Treat warps as the irreducible units, not threads.  A warp is a group
  // of 32 threads.  Threads 0-31 are, by convention, warp 0.  Threads
  // 32-63 are warp 1, and so on.  The thread's warp and lane within the
  // warp become relevant to its task.  Every thread will store these two
  // pieces of information in its registers for the duration of the kernel.
  int warpIdx = threadIdx.x / 32;
  int tgx = (threadIdx.x & 31);

  // Initialize forces within the same kernel.  Because this kernel
  // runs in only one block,
  i = threadIdx.x;
  while (i < cSh.nparticle) {
    cSh.partFrcX[i] = (float)0.0;
    cSh.partFrcY[i] = (float)0.0;
    cSh.partFrcZ[i] = (float)0.0;
    i +=  blockDim.x;
  }
  __syncthreads();
  
  //  A more advanced way, using L1 __shared__ memory
  float qq = (float)0.0;
  int nstripes = (cSh.nparticle + 31) / 32;
  int bpos = nstripes - warpIdx - 1;
  while (bpos >= 0) {

    // Read 32 particles into memory, accumulate the forces on them,
    // then write the results back to the device.  If the thread
    // would read a particle beyond the system's size, then set its
    // position as dummy numbers which will not do terrible things
    // if they get into calculations with real particles.
    //
    // NOTE HERE and BELOW: threadIdx.x = 32*warpIdx + tgx
    //
    // See note above... each thread is operating within a stripe of
    // the problem.  Accessing index threadIdx.x is integral to that.
    int prtclIdx = 32*bpos + tgx;
    if (prtclIdx < cSh.nparticle) {
      pX[threadIdx.x] = cSh.partX[prtclIdx];
      pY[threadIdx.x] = cSh.partY[prtclIdx];
      pZ[threadIdx.x] = cSh.partZ[prtclIdx];
      pQ[threadIdx.x] = cSh.partQ[prtclIdx];
    }
    else {
      pX[threadIdx.x] = (float)10000.0 + (float)(prtclIdx);
      pY[threadIdx.x] = (float)10000.0 + (float)(prtclIdx);
      pZ[threadIdx.x] = (float)10000.0 + (float)(prtclIdx);
      pQ[threadIdx.x] = (float)0.0;
    }
    
    // Loop over all particle pairs in the lower half triangle as before
    int tpos = 0;
    while (tpos <= bpos) {

      // Initialize particles as in the outer loop
      int prtclIdx = 32*tpos + tgx;
      if (prtclIdx < cSh.nparticle) {
        tX[threadIdx.x] = cSh.partX[prtclIdx];
        tY[threadIdx.x] = cSh.partY[prtclIdx];
        tZ[threadIdx.x] = cSh.partZ[prtclIdx];
        tQ[threadIdx.x] = cSh.partQ[prtclIdx];
      }
      else {

        // The offsets for particle positions must run along a different
        // (parallel, but distinct) line so that not even dummy particles
        // can ever occupy the same positions and cause a divide-by-zero.
        // As before, the charge of the dummy particles is zero.
        tX[threadIdx.x] = (float)10100.0 + (float)(prtclIdx);
        tY[threadIdx.x] = (float)10200.0 + (float)(prtclIdx);
        tZ[threadIdx.x] = (float)10300.0 + (float)(prtclIdx);
        tQ[threadIdx.x] = (float)0.0;
      }

      // Initialize tile force accumulators
      sfpX[threadIdx.x] = (float)0.0;
      sfpY[threadIdx.x] = (float)0.0;
      sfpZ[threadIdx.x] = (float)0.0;
      sftX[threadIdx.x] = (float)0.0;
      sftY[threadIdx.x] = (float)0.0;
      sftZ[threadIdx.x] = (float)0.0;

      // The tile is now ready.  Compute 32 x 32 interactions.
      // Tiles lying on the diagonal of the interaction matrix
      // will do full work for half the results.
      int imin = (bpos == tpos);
      float anti2xCountingFactor = (bpos == tpos) ? (float)0.5 : (float)1.0;
      for (i = imin; i < 32; i++) {
        int j = tgx + i;

	// Wrap j back so that it stays within the range [0, 32)
        j -= (j >= 32) * 32;

        // The value in position threadIdx.x of each __shared__
        // memory array will now be compared to one of 32 other
        // values from the array, in the range:
        // [ (threadIdx.x / 32) * 32 :: ((threadIdx.x + 31) / 32) * 32 )
        float dx    = tX[warpIdx*32 + j] - pX[threadIdx.x];
        float dy    = tY[warpIdx*32 + j] - pY[threadIdx.x];
        float dz    = tZ[warpIdx*32 + j] - pZ[threadIdx.x];
        float r2    = dx*dx + dy*dy + dz*dz;
        float r     = sqrt(r2);
        float qfac  = anti2xCountingFactor *
                      tQ[warpIdx*32 + j] * pQ[threadIdx.x];
        qq         += qfac / sqrt(r2);
        
        // This works because threadIdx.x is the only thread that will
        // ever contribute to sfpX, and the tile is arranged so that,
        // for a synchronized warp, only one thread will have a
        // contribution to make to each element of sftX.
	float fmag = qfac / (r2 * r);
        sfpX[threadIdx.x   ] += dx * fmag;
        sftX[warpIdx*32 + j] -= dx * fmag;
        sfpY[threadIdx.x   ] += dy * fmag;
        sftY[warpIdx*32 + j] -= dy * fmag;
        sfpZ[threadIdx.x   ] += dz * fmag;
        sftZ[warpIdx*32 + j] -= dz * fmag;
        __syncwarp();
      }

      // Contribute the tile force accumulations atomically to global memory
      // (DRAM).  This is only about 2x slower than atomic accumulation to
      // __shared__.  Accumulating things like this atomically to __shared__
      // would make the kernel run only about 30% slower than accumulating
      // them in an unsafe manner, willy-nilly.  Fast atomics to global are
      // a tremendous accomplishment by NVIDIA engineers!
      //
      // Note, the correspondence between 32*bpos + tgx or 32*tpos + tgx
      // and 32*warpIdx + tgx.  32*warpIdx + tgx is, again, threadIdx.x.
      atomicAdd(&cSh.partFrcX[32*bpos + tgx], sfpX[threadIdx.x]);
      atomicAdd(&cSh.partFrcY[32*bpos + tgx], sfpY[threadIdx.x]);
      atomicAdd(&cSh.partFrcZ[32*bpos + tgx], sfpZ[threadIdx.x]);
      atomicAdd(&cSh.partFrcX[32*tpos + tgx], sftX[threadIdx.x]);
      atomicAdd(&cSh.partFrcY[32*tpos + tgx], sftY[threadIdx.x]);
      atomicAdd(&cSh.partFrcZ[32*tpos + tgx], sftZ[threadIdx.x]);

      // Increment the tile counter
      tpos++;
    }

    // Increment stripe counter
    bpos -= blockDim.x / 32;
  }

  // Need to synchronize warps here as the next instructions will burn sfpX
  __syncwarp();
  
  // Reduce the energy contributions using __shared__.  This cannibalizes
  // the sfpX force accumulator, which is no longer needed.  Then make a
  // final contribution to the global array from only one thread per warp.
  // This is another global memory traffic jam mitigation.
  sfpX[threadIdx.x] = qq;
  for (i = 16; i >= 1; i /= 2) {
    if (tgx < i) {
      sfpX[threadIdx.x] += sfpX[threadIdx.x + i];
    }
    __syncwarp();
  }
  if (tgx == 0) {
    atomicAdd(&cSh.Etot[0], sfpX[threadIdx.x]);
  }
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
  gpuFloat particleXfrc, particleYfrc, particleZfrc;
  gpuFloat etot;
  
  // Create a small array of particles and populate it
  particleTypes  = CreateGpuInt(100000, 1);
  particleXcoord = CreateGpuFloat(100000, 1);
  particleYcoord = CreateGpuFloat(100000, 1);
  particleZcoord = CreateGpuFloat(100000, 1);
  particleXfrc   = CreateGpuFloat(100000, 1);
  particleYfrc   = CreateGpuFloat(100000, 1);
  particleZfrc   = CreateGpuFloat(100000, 1);
  particleCharge = CreateGpuFloat(100000, 1);

  // Allocate and initialize the total energy
  // accumulator on the host and on the device.
  etot = CreateGpuFloat(1, 1);
  
  // Initialize random number generator.  srand() SEEDS the generator,
  // thereafter each call to rand() will return a different number.
  // This is a reeally bad generator (much better methods with longer
  // periods before they start looping back over the same sequence are
  // available).
  srand(62052);
  
  // Place many, many particles
  np = 97913;
  for (i = 0; i < np; i++) {

    // Integer truncation would happen anyway, I'm just making it explicit
    particleTypes.HostData[i] = (int)(8 * rand());

    // Create some random coordinates (double-to-float conversion
    // is happening here.  On the GPU this can have performance
    // impact, so keep an eye on the data types at all times!
    particleXcoord.HostData[i] = 200.0 * (double)rand() / (double)RAND_MAX;
    particleYcoord.HostData[i] = 200.0 * (double)rand() / (double)RAND_MAX;
    particleZcoord.HostData[i] = 200.0 * (double)rand() / (double)RAND_MAX;
    particleCharge.HostData[i] = 0.5 - rand() / (double)RAND_MAX;
  }

  // CHECK
#if 0
  int j;
  double qq = 0.0;
  for (i = 0; i < np; i++) {
    for (j = 0; j < i; j++) {
      double dx = particleXcoord.HostData[i] - particleXcoord.HostData[j];
      double dy = particleYcoord.HostData[i] - particleYcoord.HostData[j];
      double dz = particleZcoord.HostData[i] - particleZcoord.HostData[j];
      double qfac = particleCharge.HostData[i] * particleCharge.HostData[j];
      qq += qfac / sqrt(dx*dx + dy*dy + dz*dz);
    }
  }
  printf("CPU result = %9.4lf\n", qq);
#endif
  // END CHECK
  
  // Stage critical constants--see cribSheet struct instance cSh above.
  cribSheet cnstage;
  cnstage.nparticle = np;
  cnstage.partX    = particleXcoord.DevcData;
  cnstage.partY    = particleYcoord.DevcData;
  cnstage.partZ    = particleZcoord.DevcData;
  cnstage.partFrcX = particleXfrc.DevcData;
  cnstage.partFrcY = particleYfrc.DevcData;
  cnstage.partFrcZ = particleZfrc.DevcData;
  cnstage.partQ    = particleCharge.DevcData;
  cnstage.Etot     = etot.DevcData; 
  
  // Upload all data to the device--note that forces are not getting
  // uploaded, as the memory is already allocated.  The forces will
  // be initialized and computed on the device.
  UploadGpuInt(&particleTypes);
  UploadGpuFloat(&particleXcoord);
  UploadGpuFloat(&particleYcoord);
  UploadGpuFloat(&particleZcoord);
  UploadGpuFloat(&particleCharge);

  // Upload the constants to the constants cache
  cudaMemcpyToSymbol(cSh, &cnstage, sizeof(cribSheet));  
  
  // Initialize energy and forces
  etot.HostData[0] = 0.0;
  UploadGpuFloat(&etot);
  ParticleSimulator<<<1, 512>>>();
  
  // Download the total energy
  DownloadGpuFloat(&etot);
  printf("Total energy (%4d threads) = %10.4f\n", 512, etot.HostData[0]);
  
  // Device synchronization
  cudaDeviceSynchronize();
  
  return 0;
}

#include <iostream>
#include <cstdio>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

using FLOAT = double;

const int NTHR_PER_BLK = 1024; // Number of CUDA threads per block
const int NBLOCK  = 56;  // Number of CUDA blocks (SMs on P100)

#define CHECK(test) if (test != cudaSuccess) throw "error";

const int Npoint = NBLOCK*NTHR_PER_BLK; // No. of independent samples
const int Neq = 100000;          // No. of generations to equilibrate 

const int Ngen_per_block = 5000; // No. of generations per block
const int Nsample = 100;         // No. of blocks to sample

const float delta = 2.0;        // Random step size

// Explicitly typed constants so can easily work with both floats and floats
static const FLOAT four = 4.0; 
static const FLOAT two  = 2.0;
static const FLOAT one  = 1.0;
static const FLOAT half = 0.5;
static const FLOAT zero = 0.0;

// Specialized/overloaded functions to support both double and float
template <typename T> __device__ __forceinline__ T rand(curandState_t* state);

template<> __device__ __forceinline__ float rand<float>(curandState_t* state) {
  return curand_uniform(state);
}

template<> __device__ __forceinline__ double rand<double>(curandState_t* state) {
  return curand_uniform_double(state);
}

__device__ __forceinline__ float EXP(float x) {return expf(x);}
__device__ __forceinline__ double EXP(double x) {return exp(x);}
__device__ __forceinline__ float SQRT(float x) {return sqrtf(x);}
__device__ __forceinline__ double SQRT(double x) {return sqrt(x);}


__global__ void SumWithinBlocks(int n, const FLOAT* data, FLOAT* blocksums) {
  int nthread = blockDim.x*gridDim.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ FLOAT sdata[2048];  // max threads?

  // Every thread in every block computes partial sum over rest of vector
  FLOAT st=zero;
  while (i < n) {
    st += data[i];
    i+=nthread;
  }
  sdata[threadIdx.x] = st;
  __syncthreads();

  // Now do binary tree sum within a block
  
  // Round up to closest power of 2
  int pow2 = 1 << (32 - __clz(blockDim.x-1));
  
  int tid = threadIdx.x;
  for (unsigned int s=pow2>>1; s>0; s>>=1) {
    if (tid<s && (tid+s)<blockDim.x) {
      //printf("%4d : %4d %4d\n", tid, s, tid+s);
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid==0) blocksums[blockIdx.x] = sdata[0];
}

void sum_stats(int Npoint, const FLOAT* stats, FLOAT* statsum, FLOAT* blocksums) {
  for (int what=0; what<4; what++) {
    SumWithinBlocks<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats+what*Npoint, blocksums);
    SumWithinBlocks<<<1,NBLOCK>>>(NBLOCK, blocksums, statsum+what);
  }
}

__device__ __forceinline__ void compute_distances(FLOAT x1, FLOAT y1, FLOAT z1, FLOAT x2, FLOAT y2, FLOAT z2,
		       FLOAT& r1, FLOAT& r2, FLOAT& r12) {
    r1 = SQRT(x1*x1 + y1*y1 + z1*z1);
    r2 = SQRT(x2*x2 + y2*y2 + z2*z2);
    FLOAT xx = x1-x2;
    FLOAT yy = y1-y2;
    FLOAT zz = z1-z2;
    r12 = SQRT(xx*xx + yy*yy + zz*zz);
}

__device__  __forceinline__ FLOAT psi(FLOAT x1, FLOAT y1, FLOAT z1, FLOAT x2, FLOAT y2, FLOAT z2) {
    FLOAT r1, r2, r12;
    compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);

    return (one + half*r12)*EXP(-two*(r1 + r2));
}

// Initialize random number generator
__global__ void initran(unsigned int seed, curandState_t* states) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  curand_init(seed, i, 0, &states[i]); // ?? correct ??
}

// zero stats counters on the GPU
__global__ void zero_stats(int Npoint, FLOAT* stats) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<Npoint) {
    stats[0*Npoint+i] = zero; // r1
    stats[1*Npoint+i] = zero; // r2
    stats[2*Npoint+i] = zero; // r12
    stats[3*Npoint+i] = zero; // accept count
  }
}

// initializes samples
__global__ void initialize(int Npoint, FLOAT* x1, FLOAT* y1, FLOAT* z1, FLOAT* x2, FLOAT* y2, FLOAT* z2, FLOAT* psir, curandState_t* states) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<Npoint) {
    x1[i] = (rand<FLOAT>(states+i) - half)*four;
    y1[i] = (rand<FLOAT>(states+i) - half)*four;
    z1[i] = (rand<FLOAT>(states+i) - half)*four;
    x2[i] = (rand<FLOAT>(states+i) - half)*four;
    y2[i] = (rand<FLOAT>(states+i) - half)*four;
    z2[i] = (rand<FLOAT>(states+i) - half)*four;
    psir[i] = psi(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]);
  }
}

__global__ void propagate(int Npoint, int nstep, FLOAT* X1, FLOAT* Y1, FLOAT* Z1, FLOAT* X2, FLOAT* Y2, FLOAT* Z2, FLOAT* P, FLOAT* stats, curandState_t* states) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<Npoint) {
    FLOAT x1 = X1[i];
    FLOAT y1 = Y1[i];
    FLOAT z1 = Z1[i];
    FLOAT x2 = X2[i];
    FLOAT y2 = Y2[i];
    FLOAT z2 = Z2[i];
    FLOAT p = P[i];
    
    for (int step=0; step<nstep; step++) {
      FLOAT x1new = x1 + (rand<FLOAT>(states+i)-half)*delta;
      FLOAT y1new = y1 + (rand<FLOAT>(states+i)-half)*delta;
      FLOAT z1new = z1 + (rand<FLOAT>(states+i)-half)*delta;
      FLOAT x2new = x2 + (rand<FLOAT>(states+i)-half)*delta;
      FLOAT y2new = y2 + (rand<FLOAT>(states+i)-half)*delta;
      FLOAT z2new = z2 + (rand<FLOAT>(states+i)-half)*delta;
      FLOAT pnew = psi(x1new, y1new, z1new, x2new, y2new, z2new);

      // This floating point mask seems about the same speed as the divergent code
      // FLOAT mask = (pnew*pnew > p*p*rand<FLOAT>(states+i)); // 1 if accept, 0 otherwise
      // FLOAT cmask = one - mask;
      // stats[3*Npoint+i] += mask;
      // p = pnew*mask + p*cmask;
      // x1 = x1new*mask + x1*cmask;
      // y1 = y1new*mask + y1*cmask;
      // z1 = z1new*mask + z1*cmask;
      // x2 = x2new*mask + x2*cmask;
      // y2 = y2new*mask + y2*cmask;
      // z2 = z2new*mask + z2*cmask;

      if (pnew*pnew > p*p*rand<FLOAT>(states+i)) {
      	stats[3*Npoint+i]++; //naccept ++;
      	p = pnew;
      	x1 = x1new;
      	y1 = y1new;
      	z1 = z1new;
      	x2 = x2new;
      	y2 = y2new;
      	z2 = z2new;
      }
      
      FLOAT r1, r2, r12;
      compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);
      
      stats[0*Npoint+i] += r1;
      stats[1*Npoint+i] += r2;
      stats[2*Npoint+i] += r12;
    }
    X1[i] = x1;  
    Y1[i] = y1;  
    Z1[i] = z1;  
    X2[i] = x2;  
    Y2[i] = y2;  
    Z2[i] = z2;  
    P[i] = p;
  }
}
  
int main() {
  FLOAT *x1, *y1, *z1, *x2, *y2, *z2, *psi, *stats, *statsum, *blocksums; // Will be allocated on the device
  curandState_t *ranstates;  

  CHECK(cudaMalloc((void **)&x1, Npoint * sizeof(FLOAT)));
  CHECK(cudaMalloc((void **)&y1, Npoint * sizeof(FLOAT)));
  CHECK(cudaMalloc((void **)&z1, Npoint * sizeof(FLOAT)));
  CHECK(cudaMalloc((void **)&x2, Npoint * sizeof(FLOAT)));
  CHECK(cudaMalloc((void **)&y2, Npoint * sizeof(FLOAT)));
  CHECK(cudaMalloc((void **)&z2, Npoint * sizeof(FLOAT)));
  CHECK(cudaMalloc((void **)&psi, Npoint * sizeof(FLOAT)));
  CHECK(cudaMalloc((void **)&stats, 4 * Npoint * sizeof(FLOAT)));
  CHECK(cudaMalloc((void **)&blocksums, NBLOCK * sizeof(FLOAT))); // workspace for summation
  CHECK(cudaMalloc((void **)&statsum, 4 * sizeof(FLOAT))); // workspace for summation
  CHECK(cudaMalloc((void **)&ranstates, Npoint*sizeof(curandState_t)));
  
  initran<<<NBLOCK,NTHR_PER_BLK>>>(5551212, ranstates);
  initialize<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, x1, y1, z1, x2, y2, z2, psi, ranstates);
  zero_stats<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats);
    
  // Equilibrate
  propagate<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, Neq, x1, y1, z1, x2, y2, z2, psi, stats, ranstates);

  // Accumulators for averages over blocks --- use doubles
  double r1_tot = zero,  r1_sq_tot = zero;
  double r2_tot = zero,  r2_sq_tot = zero;
  double r12_tot = zero, r12_sq_tot = zero;
  double naccept = zero;  // Keeps track of propagation efficiency
  for (int sample=0; sample<Nsample; sample++) {
    zero_stats<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats);
    propagate<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, Ngen_per_block, x1, y1, z1, x2, y2, z2, psi, stats, ranstates);
    
    struct {FLOAT r1, r2, r12, accept;} s;
    sum_stats(Npoint, stats, statsum, blocksums);
    CHECK(cudaMemcpy(&s, statsum, sizeof(s), cudaMemcpyDeviceToHost));

    naccept += s.accept;
    s.r1 /= Ngen_per_block*Npoint;  
    s.r2 /= Ngen_per_block*Npoint;  
    s.r12 /= Ngen_per_block*Npoint;

    printf(" block %6d  %.6f  %.6f  %.6f\n", sample, s.r1, s.r2, s.r12);

    r1_tot += s.r1;   r1_sq_tot += s.r1*s.r1;
    r2_tot += s.r2;   r2_sq_tot += s.r2*s.r2;
    r12_tot += s.r12; r12_sq_tot += s.r12*s.r12;
  }

  r1_tot /= Nsample; r1_sq_tot /= Nsample; 
  r2_tot /= Nsample; r2_sq_tot /= Nsample; 
  r12_tot /= Nsample; r12_sq_tot /= Nsample; 
  
  double r1s = sqrt((r1_sq_tot - r1_tot*r1_tot) / Nsample);
  double r2s = sqrt((r2_sq_tot - r2_tot*r2_tot) / Nsample);
  double r12s = sqrt((r12_sq_tot - r12_tot*r12_tot) / Nsample);
  
  printf(" <r1>  = %.6f +- %.6f\n", r1_tot, r1s);
  printf(" <r2>  = %.6f +- %.6f\n", r2_tot, r2s);
  printf(" <r12> = %.6f +- %.6f\n", r12_tot, r12s);
  
  printf(" acceptance ratio=%.1f%%\n",100.0*naccept/double(Npoint)/double(Ngen_per_block)/double(Nsample)); // avoid int overflow

  cudaDeviceSynchronize();  
  return 0;
}




    

    




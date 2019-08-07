#include <iostream>
#include <cstdio>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

const int NTHR_PER_BLK = 512; // Number of CUDA threads per block
const int NBLOCK  = 56;  // Number of CUDA blocks (SMs on P100)

#define CHECK(test) if (test != cudaSuccess) throw "error";

const int Npoint = NBLOCK*NTHR_PER_BLK; // No. of independent samples
const int Neq = 100000;          // No. of generations to equilibrate 

const int Ngen_per_block = 5000; // No. of generations per block
const int Nsample = 100;         // No. of blocks to sample

const double delta = 2.0;        // Random step size

__global__ void SumWithinBlocks(int n, const double* data, double* blocksums) {
  int nthread = blockDim.x*gridDim.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ double sdata[2048];  // max threads?

  // Every thread in every block computes partial sum over rest of vector
  double st=0.0;
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

void sum_stats(int Npoint, const double* stats, double* statsum, double* blocksums) {
  for (int what=0; what<4; what++) {
    SumWithinBlocks<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats+what*Npoint, blocksums);
    SumWithinBlocks<<<1,NBLOCK>>>(NBLOCK, blocksums, statsum+what);
  }
}

__device__ __forceinline__ void compute_distances(double x1, double y1, double z1, double x2, double y2, double z2,
		       double& r1, double& r2, double& r12) {
    r1 = sqrt(x1*x1 + y1*y1 + z1*z1);
    r2 = sqrt(x2*x2 + y2*y2 + z2*z2);
    double xx = x1-x2;
    double yy = y1-y2;
    double zz = z1-z2;
    r12 = sqrt(xx*xx + yy*yy + zz*zz);
}

__device__  __forceinline__ double psi(double x1, double y1, double z1, double x2, double y2, double z2) {
    double r1, r2, r12;
    compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);

    return (1.0 + 0.5*r12)*exp(-2.0*(r1 + r2));
}

// Initialize random number generator
__global__ void initran(unsigned int seed, curandState_t* states) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  curand_init(seed, i, 0, &states[i]); // ?? correct ??
}

// zero stats counters on the GPU
__global__ void zero_stats(int Npoint, double* stats) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<Npoint) {
    stats[0*Npoint+i] = 0.0; // r1
    stats[1*Npoint+i] = 0.0; // r2
    stats[2*Npoint+i] = 0.0; // r12
    stats[3*Npoint+i] = 0.0; // accept count
  }
}

// initializes samples
__global__ void initialize(int Npoint, double* x1, double* y1, double* z1, double* x2, double* y2, double* z2, double* psir, curandState_t* states) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<Npoint) {
    x1[i] = (curand_uniform_double(states+i) - 0.5)*4.0;
    y1[i] = (curand_uniform_double(states+i) - 0.5)*4.0;
    z1[i] = (curand_uniform_double(states+i) - 0.5)*4.0;
    x2[i] = (curand_uniform_double(states+i) - 0.5)*4.0;
    y2[i] = (curand_uniform_double(states+i) - 0.5)*4.0;
    z2[i] = (curand_uniform_double(states+i) - 0.5)*4.0;
    psir[i] = psi(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]);
  }
}

__global__ void propagate(int Npoint, int nstep, double* x1, double* y1, double* z1, double* x2, double* y2, double* z2, double* psir, double* stats, curandState_t* states) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<Npoint) {
    for (int step=0; step<nstep; step++) {
      double x1new = x1[i] + (curand_uniform_double(states+i)-0.5)*delta;
      double y1new = y1[i] + (curand_uniform_double(states+i)-0.5)*delta;
      double z1new = z1[i] + (curand_uniform_double(states+i)-0.5)*delta;
      double x2new = x2[i] + (curand_uniform_double(states+i)-0.5)*delta;
      double y2new = y2[i] + (curand_uniform_double(states+i)-0.5)*delta;
      double z2new = z2[i] + (curand_uniform_double(states+i)-0.5)*delta;
      double psinew = psi(x1new, y1new, z1new, x2new, y2new, z2new);
      
      if (psinew*psinew > psir[i]*psir[i]*curand_uniform_double(states+i)) {
	stats[3*Npoint+i]++; //naccept ++;
	psir[i] = psinew;
	x1[i] = x1new;
	y1[i] = y1new;
	z1[i] = z1new;
	x2[i] = x2new;
	y2[i] = y2new;
	z2[i] = z2new;
      }
      
      double r1, r2, r12;
      compute_distances(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i], r1, r2, r12);
      
      stats[0*Npoint+i] += r1;
      stats[1*Npoint+i] += r2;
      stats[2*Npoint+i] += r12;
    }
  }
}
  
int main() {
  double *x1, *y1, *z1, *x2, *y2, *z2, *psi, *stats, *statsum, *blocksums; // Will be allocated on the device
  curandState_t *ranstates;  

  CHECK(cudaMalloc((void **)&x1, Npoint * sizeof(double)));
  CHECK(cudaMalloc((void **)&y1, Npoint * sizeof(double)));
  CHECK(cudaMalloc((void **)&z1, Npoint * sizeof(double)));
  CHECK(cudaMalloc((void **)&x2, Npoint * sizeof(double)));
  CHECK(cudaMalloc((void **)&y2, Npoint * sizeof(double)));
  CHECK(cudaMalloc((void **)&z2, Npoint * sizeof(double)));
  CHECK(cudaMalloc((void **)&psi, Npoint * sizeof(double)));
  CHECK(cudaMalloc((void **)&stats, 4 * Npoint * sizeof(double)));
  CHECK(cudaMalloc((void **)&blocksums, NBLOCK * sizeof(double))); // workspace for summation
  CHECK(cudaMalloc((void **)&statsum, 4 * sizeof(double))); // workspace for summation
  CHECK(cudaMalloc((void **)&ranstates, Npoint*sizeof(curandState_t)));
  
  initran<<<NBLOCK,NTHR_PER_BLK>>>(5551212, ranstates);
  initialize<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, x1, y1, z1, x2, y2, z2, psi, ranstates);
  zero_stats<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats);
    
  // Equilibrate
  propagate<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, Neq, x1, y1, z1, x2, y2, z2, psi, stats, ranstates);

  // Accumulators for averages over blocks
  double r1_tot = 0.0,  r1_sq_tot = 0.0;
  double r2_tot = 0.0,  r2_sq_tot = 0.0;
  double r12_tot = 0.0, r12_sq_tot = 0.0;
  double naccept = 0.0;  // Keeps track of propagation efficiency
  for (int sample=0; sample<Nsample; sample++) {
    zero_stats<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, stats);
    propagate<<<NBLOCK,NTHR_PER_BLK>>>(Npoint, Ngen_per_block, x1, y1, z1, x2, y2, z2, psi, stats, ranstates);
    
    struct {double r1, r2, r12, accept;} s;
    sum_stats(Npoint, stats, statsum, blocksums);
    CHECK(cudaMemcpy(&s, statsum, sizeof(s), cudaMemcpyDeviceToHost));

    naccept += s.accept;
    s.r1 /= Ngen_per_block*Npoint;  
    s.r2 /= Ngen_per_block*Npoint;  
    s.r12 /= Ngen_per_block*Npoint;

    printf(" block %6d  %.6f  %.6f  %.6f  %.2e\n", sample, s.r1, s.r2, s.r12,s.accept);

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




    

    




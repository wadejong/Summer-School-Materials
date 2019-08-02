// In this version rather than emphasizing what can the compiler do
// with only essential modifications from us and trying to maintain
// portable, readable source code, now we see what we can do to get
// even greater performance without constraints.

// Merging scaling into vrand used to save 1 cycle but no longer does.

// Calling vdexp saves nothing (good news!).

// Relaxing accuracy of vdexp saves maybe a cycle on avx2 (on sse2 it saved several cycles)

// Tiling the loop to improve cache locality did not help (it did in
// previous versions tested on other machines that I suspect had a
// smaller L1 cache).

// Now running at ~7.6 cycles/sample on my laptop (~3.8 on sn-mem)


#include <cmath> // for exp
#include <iostream> // for cout, endl
#include <cstdlib> // for random
#include "timerstuff.h" // for cycle_count

#include <mkl_vsl.h> // for the random number generators
#include <mkl_vml.h> // for the vectorized exponential

const int NWARM = 10000;  // Number of iterations to equilbrate (aka warm up) population
const int NITER = 100000; // Number of iterations to sample
const int N = 1024;     // Population size (tried making smaller to improve caching, but no significant effect?)

double drand() {
    const double fac = 1.0/(RAND_MAX-1.0);
    return fac*random();
}

void kernel(double& x, double& p, double ran1, double ran2) {
    double xnew = ran1*23.0;
    double pnew = std::exp(-xnew);
    if (pnew > ran2*p) {
        x = xnew;
        p = pnew;
    }
}

VSLStreamStatePtr ranstate;

void vrand(int n, double* r, double a, double b) {
  //VSL_METHOD_DUNIFORM_STD in intel 14??
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ranstate, n, r, a, b);
}

int main() {
    double x[N], p[N], r[2*N], vxnew[N], vpnew[N];

    //vmlSetMode(VML_EP);
    vmlSetMode(VML_LA);
    //vmlSetMode(VML_HA);
    vslNewStream( &ranstate, VSL_BRNG_MT19937, 328409121);

    // Initialize the points
    for (int i=0; i<N; i++) {
        x[i] = drand()*23.0;
        p[i] = std::exp(-x[i]);
    }
    
    std::cout << "Equilbrating ..." << std::endl;
    for (int iter=0; iter<NWARM; iter++) {
        vrand(2*N, r, 0.0, 1.0);
        for (int i=0; i<N; i++) {
            kernel(x[i], p[i], r[i], r[i+N]);
        }
    }

    std::cout << "Sampling and measuring performance ..." << std::endl;
    double sum = 0.0;
    uint64_t Xstart = cycle_count();
    for (int iter=0; iter<NITER; iter++) {
        vrand(N, vxnew, -23.0, 0.0);
        vdExp(N, vxnew, vpnew);
        vrand(N, r, 0.0, 1.0);
#pragma simd reduction(+: sum)
        for (int i=0; i<N; i++) {
            if (vpnew[i] > r[i]*p[i]) {
                x[i] =-vxnew[i];
                p[i] = vpnew[i];
            }
            sum += x[i];
        }
    }
    uint64_t Xused = cycle_count() - Xstart;

    sum /= (NITER*N);
    std::cout.precision(10);
    std::cout << "the integral is " << sum << " over " << NITER*N << " points " << std::endl;

    double cyc = Xused / double(NITER*N);

    std::cout << cyc << " cycles per point " << std::endl;

    return 0;
}

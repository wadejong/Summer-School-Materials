// In this version we switch to using single precision end-to-end.
// Likely need to be a bit more careful in accumlators, though since
// all quantities being accumlated are positive and similar magnitude
// we are likely OK.  Given the large statistical error this is
// likely every bit as accurate as the double precision version.

// Now running in 3.6 cyles/sample on my laptop (~2.02 on sn-mem) --- 20 faster than the original!

// Why is it almost exactly 2x faster than the d.p. version?

#include <cmath> // for exp
#include <iostream> // for cout, endl
#include <cstdlib> // for random
#include "timerstuff.h" // for cycle_count

#include <mkl_vsl.h> // for the random number generators
#include <mkl_vml.h> // for the vectorized exponential

const int NWARM = 10000;  // Number of iterations to equilbrate (aka warm up) population
const int NITER = 100000; // Number of iterations to sample
const int N = 1024;     // Population size (tried making smaller to improve caching, but no significant effect?)

float srand() {
    const float fac = 1.0/(RAND_MAX-1.0);
    return fac*random();
}

void kernel(float& x, float& p, float ran1, float ran2) {
    float xnew = ran1*23.0;
    float pnew = std::exp(-xnew);
    if (pnew > ran2*p) {
        x = xnew;
        p = pnew;
    }
}

VSLStreamStatePtr ranstate;

void vrand(int n, float* r, float a, float b) {
  //VSL_METHOD_DUNIFORM_STD in intel 14??
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ranstate, n, r, a, b);
}

int main() {
    float x[N], p[N], r[2*N], vxnew[N], vpnew[N];

    //vmlSetMode(VML_EP);
    vmlSetMode(VML_LA);
    //vmlSetMode(VML_HA);
    vslNewStream( &ranstate, VSL_BRNG_MT19937, 328409121);

    // Initialize the points
    for (int i=0; i<N; i++) {
        x[i] = srand()*23.0;
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
    float sum = 0.0;
    uint64_t Xstart = cycle_count();
    for (int iter=0; iter<NITER; iter++) {
        vrand(N, vxnew, -23.0, 0.0);
        vsExp(N, vxnew, vpnew);
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

    float cyc = Xused / float(NITER*N);

    std::cout << cyc << " cycles per point " << std::endl;

    return 0;
}

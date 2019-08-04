#include <math.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <algorithm>
#include "timerstuff.h"

using namespace std;

double test(int n, double a, double * __restrict__ x, double * __restrict__ y) {
    uint64_t usedmin = 99999999999;
    int niter = std::max(3, 40000/n);

    for (int attempt=0; attempt<4; attempt++) {
        uint64_t start = cycle_count();
        for (int iter=0; iter<niter; iter++) {

	  cblas_daxpy(n, a, x, 1, y, 1);

	  //for (int i=0; i<n; i++) {
	  //                  y[i] += a*x[i];
	  //}

            y[1] += 0.1; // force optimizer to not reorder loops
        }
        uint64_t used = cycle_count() - start;
        if (attempt>0) usedmin = std::min(used,usedmin);
    }
    return usedmin/double(n)/double(niter);
}


int main() {
    const int NMAX = 1024*1024*128;
    double a = 1.1;
    alignas(64) double x[NMAX];
    alignas(64) double y[NMAX];
    
    for (int i=0; i<NMAX; i++) {
        x[i] = 1.0;
    }
    for (int i=0; i<NMAX; i++) {
        y[i] = 0.0;
    }

    for (int n=1; n<1024; n+=1) {
        std::cout << n << " " << test(n,a,x,y) << std::endl;
    }

    for (int n=1024; n<131072; n+=128) {
        std::cout << n << " " << test(n,a,x,y) << std::endl;
    }

    for (int n=131072; n<=NMAX; n*=2) {
        std::cout << n << " " << test(n,a,x,y) << std::endl;
    }

    return 0;
}

    

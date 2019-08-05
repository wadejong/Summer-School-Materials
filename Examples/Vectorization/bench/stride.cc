#include <math.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <algorithm>
#include "timerstuff.h"

using namespace std;

double test(int n, double a, double * __restrict__ x, double * __restrict__ y, int stride) {
  uint64_t usedmin = 99999999999;
  for (int attempt=0; attempt<4; attempt++) {
    uint64_t start = cycle_count();
    
    cblas_daxpy(n/stride, a, x, stride, y, stride);
    
    //for (int i=0; i<n; i+=stride) {
    //   y[i] += a*x[i];
    //}
    
    uint64_t used = cycle_count() - start;
    if (attempt>0) usedmin = std::min(used,usedmin);
    
    y[1] += 0.1; // force optimizer to not get clever
  }
  
  return usedmin/double(n/stride);
}


int main() {
    const int NMAX = 1024*1024*2047;
    double a = 1.1;
    alignas(64) double x[NMAX];
    alignas(64) double y[NMAX];
    
    for (int i=0; i<NMAX; i++) {
        x[i] = 1.0;
    }
    for (int i=0; i<NMAX; i++) {
        y[i] = 0.0;
    }

    int strides[] = {1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,32,64,96,128,160,196,256,512,1024,2048,4096};

    for (int stride : strides) {
      int ndo = std::min(NMAX/stride,1024*1024*15)*stride;
      std::cout << stride << " " << test(ndo,a,x,y,stride) << std::endl;
    }

    return 0;
}

    

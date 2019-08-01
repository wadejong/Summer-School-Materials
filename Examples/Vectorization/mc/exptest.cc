// This program measures the speed of exp() using arguments that span
// a wide range of values

#include <math.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <mkl_vml.h>
#include <float.h>
#include <algorithm>
#include "timerstuff.h"

using namespace std;

const int buflen = 1024; // fits easily in L1

uint64_t cycles_used_single_vml = 0;
uint64_t cycles_used_double_vml = 0;
uint64_t cycles_used_single = 0;
uint64_t cycles_used_double = 0;

void vsExpX(int n, const float* sx, float* sr) {
    for (int i=0; i<n; i++) sr[i] += expf(sx[i]);
}

void vdExpX(int n, const double* sx, double* sr) {
    for (int i=0; i<n; i++) sr[i] += exp(sx[i]);
}

void vExp(int n, const double* dx, double* dr) {
    vdExp(n, dx, dr);
}

void vExp(int n, const float* sx, float* sr) {
    vsExp(n, sx, sr);
}

void test(int n, const float* sx) {
    float sr[buflen]  __attribute__ ((aligned (128)));
    double dx[buflen] __attribute__ ((aligned (128)));
    double dr[buflen]  __attribute__ ((aligned (128)));

    // Time Intel VML single precision
    uint64_t start = cycle_count();
    vsExp(n, sx, sr);
    cycles_used_single_vml += cycle_count() - start;

    // Time compiler single precision
    start = cycle_count();
    vsExpX(n, sx, sr);
    cycles_used_single += cycle_count() - start;
    
    // Generate a double precision version of x
    for (int i=0; i<buflen; i++) dx[i] = sx[i];

    // Time Intel VML double precision
    start = cycle_count();
    vdExp(n, dx, dr);
    cycles_used_double_vml += cycle_count() - start;

    // Time compiler double precision
    start = cycle_count();
    vdExpX(n, dx, dr);
    cycles_used_double += cycle_count() - start;
}


int main() {
    const float log10_float_max = float(log10(double(FLT_MAX)));
    const float log10_float_min = float(log10(double(FLT_MIN)));

    //vmlSetMode(VML_EP); // half of the signifcand bits could be wrong!
    //vmlSetMode(VML_LA); // only 4 ulp error
    vmlSetMode(VML_HA); // only 1 ulp error

    printf("      float max = %.8e  min = %.8e\n", FLT_MAX, FLT_MIN);
    printf("log10 float max = %.8e  min = %.8e\n", log10_float_max, log10_float_min);


    float xbuf[buflen] __attribute__ ((aligned (16))); //  Buffer for testing

    unsigned int ndone = 0; // Counter for values tested
    int n = 0;// Counter for elements in the buffer

    // Loop thru all possible 32-bit words collecting them in batches of buflen
    unsigned int i = 0;
    do {
        // Interpret the word as a float
        float value = *((float*)(&i));

        // Check that its value is "sensible"
        if (isnormal(value) && value<log10_float_max && value>log10_float_min) {
            xbuf[n++] = value;
            ndone++;

            // If the buffer is full then test
            if (n == buflen) {
                test(n, xbuf);
                n = 0;
            }
        }
        i++;
    } while (i);

    printf("ndone %u\n", ndone);
    printf("single precision cycles per element vml      %.1f\n", double(cycles_used_single_vml)/ndone);
    printf("single precision cycles per element compiler %.1f\n", double(cycles_used_single)/ndone);
    printf("double precision cycles per element vml      %.1f\n", double(cycles_used_double_vml)/ndone);
    printf("double precision cycles per element compiler %.1f\n", double(cycles_used_double)/ndone);

    return 0;
}

    

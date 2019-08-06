#include <iostream>
#include <cstdio>
#include <cmath>
#include <mkl_vsl.h> // for the random number generators
#include <mkl_vml.h> // for the vectorized exponential
#include <vector>

using namespace std;

VSLStreamStatePtr ranstate;

void vrand(int n, double* r) {
  //VSL_METHOD_DUNIFORM_STD in intel 14??
  vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, ranstate, n, r, 0.0, 1.0);
}

const int Npoint = 1000;         // No. of independent samples
const int Neq = 100000;          // No. of generations to equilibrate 
const int Ngen_per_block = 5000; // No. of generations per block
const int Nsample = 100;         // No. of blocks to sample

const double delta = 2.0;        // Random step size

long naccept = 0;                // Keeps track of propagation efficiency
long nreject = 0;

void compute_distances(double x1, double y1, double z1, double x2, double y2, double z2,
		       double& r1, double& r2, double& r12) {
    r1 = sqrt(x1*x1 + y1*y1 + z1*z1);
    r2 = sqrt(x2*x2 + y2*y2 + z2*z2);
    double xx = x1-x2;
    double yy = y1-y2;
    double zz = z1-z2;
    r12 = sqrt(xx*xx + yy*yy + zz*zz);
}

double psi(double x1, double y1, double z1, double x2, double y2, double z2) {
    double r1, r2, r12;
    compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);

    return (1.0 + 0.5*r12)*exp(-2.0*(r1 + r2));
}

// initializes samples
void initialize(double ran1, double ran2, double ran3, double ran4, double ran5, double ran6,
		double& x1, double& y1, double& z1, double& x2, double& y2, double& z2, double& psir) {
  x1 = (ran1 - 0.5)*4.0;
  y1 = (ran2 - 0.5)*4.0;
  z1 = (ran3 - 0.5)*4.0;
  x2 = (ran4 - 0.5)*4.0;
  y2 = (ran5 - 0.5)*4.0;
  z2 = (ran6 - 0.5)*4.0;
  psir = psi(x1, y1, z1, x2, y2, z2);
}

void propagate(double& x1, double& y1, double& z1, double& x2, double& y2, double& z2, 
	       double& psir,
	       double ran1, double ran2, double ran3, double ran4, double ran5, double ran6, double ran7) {
  double x1new = x1 + (ran1-0.5)*delta;
  double y1new = y1 + (ran2-0.5)*delta;
  double z1new = z1 + (ran3-0.5)*delta;
  double x2new = x2 + (ran4-0.5)*delta;
  double y2new = y2 + (ran5-0.5)*delta;
  double z2new = z2 + (ran6-0.5)*delta;
  double psinew = psi(x1new, y1new, z1new, x2new, y2new, z2new);
  
  if (psinew*psinew > psir*psir*ran7) {
    naccept ++;
    psir = psinew;
    x1 = x1new;
    y1 = y1new;
    z1 = z1new;
    x2 = x2new;
    y2 = y2new;
    z2 = z2new;
  }
  else {
    nreject ++;
  }
}
  
void accumulate_stats(double x1, double y1, double z1, double x2, double y2, double z2, double& r1_block, double& r2_block, double& r12_block) {
    double r1, r2, r12;
    compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);

    r1_block += r1;  r2_block += r2;  r12_block += r12;
}

int main() {
  vector<double> x1(Npoint), y1(Npoint), z1(Npoint);
  vector<double> x2(Npoint), y2(Npoint), z2(Npoint);
  vector<double> PSI(Npoint); // Holds wave function values

  double ran[Npoint*7];
    double* ran1 = ran;
    double* ran2 = ran + Npoint;
    double* ran3 = ran + Npoint*2;
    double* ran4 = ran + Npoint*3;
    double* ran5 = ran + Npoint*4;
    double* ran6 = ran + Npoint*5;
    double* ran7 = ran + Npoint*6;

    vslNewStream( &ranstate, VSL_BRNG_MT19937, 328409121);
    
    vrand(Npoint*7, ran);
    for (int i=0; i<Npoint; i++) {
      initialize(ran1[i], ran2[i], ran3[i], ran4[i], ran5[i], ran6[i],
		 x1[i], y1[i], z1[i], x2[i], y2[i], z2[i], PSI[i]);
    }
    
    for (int step=0; step<Neq; step++) { // Equilibrate
        vrand(Npoint*7, ran);
        for (int i=0; i<Npoint; i++) {
	  propagate(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i], PSI[i],
		    ran1[i], ran2[i], ran3[i], ran4[i], ran5[i], ran6[i], ran7[i]);
        }
    }

    naccept = nreject = 0;

    // Accumulators for averages over blocks
    double r1_tot = 0.0,  r1_sq_tot = 0.0;
    double r2_tot = 0.0,  r2_sq_tot = 0.0;
    double r12_tot = 0.0, r12_sq_tot = 0.0;

    for (int block=0; block<Nsample; block++) {

        // Accumulators for averages over points in block
        double r1_block = 0.0, r2_block = 0.0, r12_block = 0.0;

        for (int step=0; step<Ngen_per_block; step++) {
	  vrand(Npoint*7, ran);
	  for (int i=0; i<Npoint; i++) {
	    propagate(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i], PSI[i],
		      ran1[i], ran2[i], ran3[i], ran4[i], ran5[i], ran6[i], ran7[i]);
	    accumulate_stats(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i], r1_block, r2_block, r12_block);
	  }
        }

        r1_block /= Ngen_per_block*Npoint;  
        r2_block /= Ngen_per_block*Npoint;  
        r12_block /= Ngen_per_block*Npoint;

        printf(" block %6d  %.6f  %.6f  %.6f\n", block, r1_block, r2_block, r12_block);

        r1_tot += r1_block;   r1_sq_tot += r1_block*r1_block;
        r2_tot += r2_block;   r2_sq_tot += r2_block*r2_block;
        r12_tot += r12_block; r12_sq_tot += r12_block*r12_block;
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

    printf(" accept=%ld    reject=%ld    acceptance ratio=%.1f%%\n", 
           naccept, nreject, 100.0*naccept/(naccept+nreject));

    return 0;
}




    

    




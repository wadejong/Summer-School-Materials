/***

This implements a master-worker model to parallelize the recursive quadrature example. 

In this 1-D example, it is unlikely master-worker decomposition will
give a a performance benefit since there is so little computation per
task compared to the amount of communication.  However, it is instructive to 
adopt this approach.

We start by diving the range [a,b] into 128 sub intervals.  These are
dynamically allocated by the master to a pool of workers.
 ***/

#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iostream>

double g(double x) {
  return exp(-x*x)*cos(3*x);
}

// 3-point Gauss-Legendre quadrature on [a,b]
double I(const double a, const double b, double (*f)(double)) {
  // Points/weights for GL on [0,1]
  const int N = 3;
  const double x[3] = {8.87298334620741702e-01, 5.00000000000000000e-01, 1.12701665379258312e-01};
  const double w[3] = {2.77777777777777790e-01, 4.44444444444444420e-01, 2.77777777777777790e-01};
  
  double L = (b-a);
  double sum = 0.0;
  for (int i=0; i<N; i++) {
    sum += w[i]*f(a+L*x[i]);
  }
  return sum*L;
}

double Irecur(const double a, const double b, double (*f)(double), const double eps, int level=0) {
  const double middle = (a+b)*0.5;
  double total=I(a,b,f);
  double left = I(a,middle,f);
  double right= I(middle,b,f);
  double test=left+right;
  double err=std::abs(total-test);
  
  //for (int i=0; i<level; i++) printf("  ");
  //printf("[%6.3f,%6.3f] total=%.6e test=%.6e err=%.2e\n", a, b, total, test, err);
  if (level >= 20) return test; // 2^20 = 1M boxes
  
  if (err<eps) 
    return test;
  else {
    double neweps = std::max(eps*0.5,1e-15*std::abs(total));
    return Irecur(a,middle,f,neweps,level+1) + Irecur(middle,b,f,neweps,level+1);
  }
}

// Information that defines a task
struct Info {
  double a;
  double b;
  int flag; // 0=work 1=die
  
  Info(double a, double b, int flag) : a(a), b(b), flag(flag) {}
  
  Info() : a(0), b(0), flag(-1) {}
};

// Made this global out of laziness
const double eps=1e-10;

// Only the master (rank=0) executes this
void master() {
  int nproc, rank, nworker;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  nworker = nproc-1;
  
  const double gexact = std::sqrt(4.0*std::atan(1.0))*std::exp(-9.0/4.0);
  const double a=-6.0, b=6.0;
  
  const int NBOX = 128;
  double result = 0.0;
  
  int nbusy=0;
  for (int box=0; box<NBOX; box++) {
    double abox = a+box*(b-a)/NBOX;
    double bbox = a+(box+1)*(b-a)/NBOX;
    
    int nextworker;
    if (nbusy < nworker) { // In startup keep assigning tasks to the next worker until everyone is busy
      nextworker = nbusy+1;
      nbusy += 1;
    }
    else { // Once everyone is working receive a result and assign more work
      double value;
      MPI_Status status;
      MPI_Recv(&value, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
      result += value;
      nextworker = status.MPI_SOURCE;
    }
    Info info(abox, bbox, 0);
    MPI_Send(&info, sizeof(info), MPI_BYTE, nextworker, 0, MPI_COMM_WORLD);      
  }
  
  // Once all work has been issued, receive any outstanding results
  for (int i=0; i<nbusy; i++) {
    double value;
    MPI_Status status;
    MPI_Recv(&value, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
    result += value;
  }
  
  // Tell the workers to finish
  for (int i=0; i<nworker; i++) {
    Info info(0, 0, 1);
    MPI_Send(&info, sizeof(info), MPI_BYTE, i+1, 0, MPI_COMM_WORLD);      
  }
  
  // Print results
  double oldresult = Irecur(a,b,g,eps);
  double err_exact = std::abs(result-gexact);
  printf("result=%.10e   old_result=%.10e err-exact=%.2e\n",
	 result, oldresult, err_exact);
}

// All workers (rank>0) execute this
void worker() {
  Info info;
  int nproc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  while (1) {
    // Wait for a message from the master
    MPI_Status status;
    MPI_Recv(&info, sizeof(info), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
    //std::cout << "Worker: " << rank << " " << info.a << " " << info.b << " " << info.flag << std::endl;
    
    if (info.flag == 0) { // Work!
      double value = Irecur(info.a, info.b, g, eps);
      MPI_Send(&value, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    else if (info.flag == 1) { // Die!
      return;
    }
    else { // Confusion
      std::cout << "Uh?";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc,&argv);
  int nproc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    master();
  }
  else {
    worker();
  }
  
  MPI_Finalize();
  return 0;
}

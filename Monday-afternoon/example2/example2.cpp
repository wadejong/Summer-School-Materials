#include <stdio.h>
#include <omp.h>
#include <math.h>

int N = 1000000000;

void main()
{

  //declare two arrays
  int* a = new int[N];
  int* b = new int[N];

  //initialize a
  for (int i=0; i<N; i++) {
    a[i] = 1.0;
  }

  //initialize b
  b[0] = 1.0;
  for (int i=1; i<N; i++) {
    b[i] = b[i-1] + 1.0;
  }

  //add the two arrays
  for (int i=0; i<N; i++) {
    a[i] = a[i] + b[i];
  }

  //average the result
  double average = 0.0;
  for (int i=0; i<N; i++) {
    average += a[i];
  }
  average = average/double(N);

  //print the result
  printf("Average: %f\n",average);

  delete(a);
  delete(b);
}

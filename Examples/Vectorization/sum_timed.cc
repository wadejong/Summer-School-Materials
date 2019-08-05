#include <iostream>
#include <algorithm>
#include "bench/timerstuff.h"

int main() {
    const int N=4096;// sized to fit into L1
    alignas(4096) double a[N]; // align with page
    for (int i=0; i<N; i++) a[i] = i;

    double sum = 0.0;
    uint64_t used = 99999999999;
    for (int repeat=0; repeat<100000; repeat++) {
      uint64_t start = cycle_count();
      for (int i=0; i<N; i++) sum += a[i];
      used = std::min(used,cycle_count() - start);
      a[1] = 1.0;
    }
    double cycles_per_element = double(used)/N;

    std::cout << sum << std::endl;
    std::cout << cycles_per_element << std::endl;

    return 0;
}

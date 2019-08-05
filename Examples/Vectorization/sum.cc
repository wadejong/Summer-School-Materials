#include <iostream>

int main() {
    const int N=100007;
    double a[N];
    __asm__("/*startloop*/");
    for (int i=0; i<N; i++) a[i] = i;
    double sum = 0.0;
    for (int i=0; i<N; i++) sum += a[i];
    __asm__("/*endloop*/");
    std::cout << sum << std::endl;
    return 0;
}

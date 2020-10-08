#include <iostream>
#include <omp.h>


int main() {
    printf("Number of Processors: %d\n", omp_get_num_procs());

    #pragma omp parallel num_threads(2) default(none)
    for (int i = 0; i < 4; ++i) {
        printf("Hello from Thread %d, i: %d \n", omp_get_thread_num(), i);
    }

    return 0;
}

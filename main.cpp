#include <iostream>
#include <omp.h>


int main() {
    printf("Number of Processors: %d\n", omp_get_num_procs());

    #pragma omp parallel num_threads(55) default(none)
    for (int i = 0; i < 400000; ++i) {
        if(i%10 == 0){
            printf("Hello from Thread %d, i: %d \n", omp_get_thread_num(), i);
        }

    }

    return 0;
}

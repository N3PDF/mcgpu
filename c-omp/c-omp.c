#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "integrands/lepage_test.h"
#include "Vegas.h"



int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage %s number_of_iterations\n", argv[0]);
        exit(0);
    }
    int n_dim = 4;
    int n_events = (int) 1e6;
    int n_iter = atoi(argv[1]);

    double res, sigma;
    struct timeval start, stop;

    gettimeofday(&start, 0);
    int state = vegas(lepage_test, 1, n_dim, n_iter, n_events, &res, &sigma);
    gettimeofday(&stop, 0);

    if (state != 0) {
       printf("Something went wrong\n");
    }

    printf("Final result: %1.5f +- %1.5f\n", res, sigma);
    double result = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*1e-6;
    printf("It took: %1.7f seconds\n", result);
}


//gettimeofday(&t0, 0);
///* ... */
//gettimeofday(&t1, 0);
//long elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;

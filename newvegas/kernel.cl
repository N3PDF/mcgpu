#include "definitions.h"

// Include opencl files
#include "integrand.cl"

void read_divisions(__global const double *arrin, const int n_dim, double arrout[MAXDIM][BINS_MAX]) {
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    for (int j = 0 ; j < n_dim; j ++ ) {
        const int idx = j*BINS_MAX;
        for (int k = 0; k < BINS_MAX; k++) {
            arrout[j][k] = arrin[idx+k];
        }
    }
}
void read_randoms(__global const double *arrin, const int n_dim, double arrout[BUFFER_SIZE][MAXDIM]) {
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    for (int j = 0 ; j < n_dim; j ++ ) {
        const int idx = j*BUFFER_SIZE;
        for (int b = 0; b < BUFFER_SIZE; b++) {
            arrout[b][j] = arrin[idx+b];
        }
    }
}

void write_indexes(const short arrin[BUFFER_SIZE*MAXDIM], const int n_dim, __global short *arrout) {
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    const int jmax = BUFFER_SIZE*n_dim;
    for (int j = 0; j < MAXDIM*BUFFER_SIZE; j++) {
        if (j >= jmax) break;
        arrout[j] = arrin[j];
    }
}

void write_results(const double arrin[BUFFER_SIZE], __global double arrout[BUFFER_SIZE]) {
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    for (int b = 0; b < BUFFER_SIZE; b++) {
        arrout[b] = arrin[b];
    }
}

#define reg_i 0.0
#define reg_f 1.0
void digest_random(const double divisions[MAXDIM][BINS_MAX], const double randoms[BUFFER_SIZE][MAXDIM],
        const int n_dim,
        double vegas_rand[BUFFER_SIZE][MAXDIM], short indexes[BUFFER_SIZE][MAXDIM], double wgts[BUFFER_SIZE]) {
    // Not clear at all in which order it is better to write these loops
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    for (int b = 0; b < BUFFER_SIZE; b++) {
        double wgt = 1.0;
        for (int j = 0; j < MAXDIM; j++) { // TODO maybe we should have a function per dimension for dataflow
            if (j >= n_dim) break;
            const double rn = randoms[b][j];
            const double xn = BINS_MAX*(1.0 - rn);
            short int_xn = (short) max(0, min( (int) xn, BINS_MAX));
            const double aux_rand = xn - int_xn;
            double x_ini = 0.0;
            if (int_xn > 0) {
                x_ini = divisions[j][int_xn-1];
            }
            const double xdelta = divisions[j][int_xn] - x_ini;
            const double rand_x = x_ini + xdelta*aux_rand;
            wgt *= xdelta*BINS_MAX;
            vegas_rand[b][j] = reg_i + rand_x*(reg_f - reg_i);
            indexes[b][j] = int_xn;
        }
        wgts[b] = wgt;
    }
}

void integrand_computer(const double randoms[BUFFER_SIZE][MAXDIM], const double wgts[BUFFER_SIZE],
        const int n_dim,
        double results[BUFFER_SIZE]) {
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    for (int b = 0; b < BUFFER_SIZE; b++) {
        double tmp = integrand(n_dim, randoms[b]);
        results[b] = wgts[b]*tmp;
    }
}



// Kernel to be run per event
__kernel
__attribute__((reqd_work_group_size(1,1,1)))
#ifdef FPGABUILD
__attribute__((xcl_dataflow))
#endif
void events_kernel(__global const double *divisions_in, __global const double *randoms_in, 
        const int n_dim,
        __global double results_out[BUFFER_SIZE], __global short *indexes_out) {

    // Step 1. Allocate local arrays to store results
    double divisions[MAXDIM][BINS_MAX];
    double randoms[BUFFER_SIZE][MAXDIM];
    double vegas_rand[BUFFER_SIZE][MAXDIM];
    double results[BUFFER_SIZE];
    double wgts[BUFFER_SIZE];
    short indexes[BUFFER_SIZE][MAXDIM];

    // Step 2. Buffer reads
    read_divisions(divisions_in, n_dim, divisions);
    read_randoms(randoms_in, n_dim, randoms);

    // Step 2. Generate vegas random numbers
    digest_random(divisions, randoms, n_dim, vegas_rand, indexes, wgts);

    // Step 3. Compute the integrand
    integrand_computer(vegas_rand, wgts, n_dim, results);

    // Step 4. Copy out the div indexex (note that this one does not depend in step 3
    // and hopefully will happen in parallel)
    write_indexes(indexes, n_dim, indexes_out);

    // Step 5. Copy out the resuts
    write_results(results, results_out);
}

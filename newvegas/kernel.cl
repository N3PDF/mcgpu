#include "definitions.h"

// Include opencl files
#include "integrand.cl"

void read_divisions(__global const double arrin[MAXDIM][BINS_MAX], double arrout[MAXDIM][BINS_MAX]) {
#ifdef FPGABUILD
    __attribute__((opencl_unroll_hint))
#endif
    for (short j = 0 ; j < MAXDIM; j ++ ) {
        for (short k = 0; k < BINS_MAX; k++) {
            arrout[j][k] = arrin[j][k];
        }
    }
}

void read_randoms(__global const double arrin[BUFFER_SIZE], const short jdim, double arrout[BUFFER_SIZE][MAXDIM]) {
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    for (short b = 0; b < BUFFER_SIZE; b++) {
        arrout[b][jdim] = arrin[b];
    }
}

void write_indexes(const short arrin[BUFFER_SIZE], __global short arrout[BUFFER_SIZE]) {
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    for(short b = 0; b < BUFFER_SIZE; b++) {
        arrout[b] = arrin[b];
    }
}

void write_results(const double arrin[BUFFER_SIZE], __global double arrout[BUFFER_SIZE]) {
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    for (short b = 0; b < BUFFER_SIZE; b++) {
        arrout[b] = arrin[b];
    }
}

#define reg_i 0.0
#define reg_f 1.0
void digest_random(const double divisions[MAXDIM][BINS_MAX], const double randoms[BUFFER_SIZE][MAXDIM],
        double vegas_rand[BUFFER_SIZE][MAXDIM], short indexes[MAXDIM][BUFFER_SIZE], double wgts[BUFFER_SIZE]) {
    // Not clear at all in which order it is better to write these loops
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    for (short b = 0; b < BUFFER_SIZE; b++) {
        double wgt = 1.0;
        for (short j = 0; j < MAXDIM; j++) { 
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
            indexes[j][b] = int_xn;
        }
        wgts[b] = wgt;
    }
}

void integrand_computer(const double randoms[BUFFER_SIZE][MAXDIM], const double wgts[BUFFER_SIZE],
        double results[BUFFER_SIZE]) {
#ifdef FPGABUILD
__attribute__((xcl_pipeline_loop(1)))
#endif
    for (short b = 0; b < BUFFER_SIZE; b++) {
        double tmp = integrand(MAXDIM, randoms[b]);
        results[b] = wgts[b]*tmp;
    }
}



// Kernel to be run per event
__kernel
__attribute__((reqd_work_group_size(1,1,1)))
#ifdef FPGABUILD
__attribute__((xcl_dataflow))
#endif
void events_kernel(__global const double divisions_in[MAXDIM][BINS_MAX], __global const double randoms_in[MAXDIM][BUFFER_SIZE], 
        __global double results_out[BUFFER_SIZE], __global short indexes_out[MAXDIM][BUFFER_SIZE]) {

    // Step 0. Allocate local arrays to store results
    double divisions[MAXDIM][BINS_MAX];
    double randoms[BUFFER_SIZE][MAXDIM];
    double vegas_rand[BUFFER_SIZE][MAXDIM];
    double results[BUFFER_SIZE];
    double wgts[BUFFER_SIZE];
    short indexes[BUFFER_SIZE][MAXDIM];

    // Step 1. Read the divisions
    read_divisions(divisions_in, divisions);

    // BUFFER-loop functions

    // Step 2. Buffer reads
    for (short j = 0; j < MAXDIM; j++) {
        read_randoms(randoms_in[j], j, randoms);
    }

    // Step 2. Generate vegas random numbers
    digest_random(divisions, randoms, vegas_rand, indexes, wgts);

    // Step 3. Compute the integrand
    integrand_computer(vegas_rand, wgts, results);

    // Step 4. Copy out the div indexex (note that this one does not depend in step 3
    // and hopefully will happen in parallel)
    for (short j = 0; j < MAXDIM; j++) {
        write_indexes(indexes[j], indexes_out[j]);
    }

    // Step 5. Copy out the resuts
    write_results(results, results_out);
}

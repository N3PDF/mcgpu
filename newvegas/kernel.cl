#include "definitions.h"

// Include opencl files
#include "random.cl"
#include "integrand.cl"

// In this approach, the number of events which are computed every time the kernel
// is called is 100% fixed
// The fix can be, later on, memory motivated, but for now
// we are going to use an arbitrary BUFFER_SIZE of 1024

// Burst read the input division array
void read_divs(__global const double *div_in, double div_out[MAXDIM][BINS_MAX], const int n_dim) {
    __attribute__((xcl_pipeline_loop(1)))
    for (int j = 0; j < n_dim; j++) {
        const int jdx = j*BINS_MAX;
        for (int k = 0; k < BINS_MAX; k++) {
            div_out[j][k] = div_in[jdx + k];
        }
    }
}

void rng_initializer(const int index, STATE_RNG rng[BUFFER_SIZE]) {
    __attribute__((xcl_pipeline_loop(1)))
    for (int i = 0; i < BUFFER_SIZE; i++) {
        const int idx = index+i;
        RGN_INITIALIZER(&rng[i], idx, idx);
    }
}

#define reg_i 0.0
#define reg_f 1.0
void random_computer(const double divisions[BINS_MAX], const double wgt_in[BUFFER_SIZE],
        STATE_RNG rng[BUFFER_SIZE],
        double randoms[BUFFER_SIZE], int indexes[BUFFER_SIZE], double wgts[BUFFER_SIZE]) {
    // hey, I just found you
    // and this is craaaazy
    // but here's my arrays
    // compile me maybe
    __attribute__((xcl_pipeline_loop(1)))
    for (int i = 0; i < BUFFER_SIZE; i++) {
        const double rn = GEN_RAN(&rng[i]);
        const double xn = BINS_MAX*(1.0 - rn);
        int int_xn = (int) max(0, min( (int) xn, BINS_MAX));
        const double aux_rand = xn - int_xn;
        double x_ini = 0.0;
        if (int_xn > 0) {
            x_ini = divisions[int_xn-1];
        }
        const double xdelta = divisions[int_xn] - x_ini;
        const double rand_x = x_ini + xdelta*aux_rand;
        wgts[i] = wgt_in[i]*xdelta*BINS_MAX;
        randoms[i] = reg_i + rand_x*(reg_f-reg_i);
        indexes[i] = int_xn;
    }
}
void initialize_1d(double arr[BUFFER_SIZE], const double val) {
    __attribute__((xcl_pipeline_loop(1)))
    for (int i = 0; i < BUFFER_SIZE; i++) {
        arr[i] = val;
    }
}
void copybuffer_1d(const double res_in[BUFFER_SIZE], double res_out[BUFFER_SIZE]) {
    __attribute__((xcl_pipeline_loop(1)))
    for (int i = 0; i < BUFFER_SIZE; i++) {
        res_out[i] = res_in[i];
    }
}
void copyrand_1d(const STATE_RNG rin[BUFFER_SIZE], STATE_RNG rout[BUFFER_SIZE]) {
    __attribute__((xcl_pipeline_loop(1)))
    for (int i = 0; i < BUFFER_SIZE; i++) {
        rout[i] = rin[i];
    }
}
void digest_random(const double randoms[MAXDIM][BUFFER_SIZE],
        const int n_dim, 
        double integrator_input[BUFFER_SIZE][MAXDIM]) {
    // For instance, compute the momentums
    for (int i = 0; i < BUFFER_SIZE; i++) {
        for (int j = 0; j < n_dim; j++) {
            integrator_input[i][j] = randoms[j][i];
        }
    }
}

void integral_computer(const double randoms[BUFFER_SIZE][MAXDIM], const double wgts[BUFFER_SIZE],
        const int n_dim,
        double all_results[BUFFER_SIZE]) {
    __attribute__((xcl_pipeline_loop(1)))
    for (int i = 0; i < BUFFER_SIZE; i++) {
        all_results[i] = wgts[i]*integrand(n_dim, randoms[i]);
    }
}

void return_1d(const double res_in[BUFFER_SIZE], __global double res_out[BUFFER_SIZE]) {
    __attribute__((xcl_pipeline_loop(1)))
    for (int i = 0; i < BUFFER_SIZE; i++) {
        res_out[i] = res_in[i];
    }
}
void return_2d(const int res_in[MAXDIM][BUFFER_SIZE], const int n_dim, __global int res_out[BUFFER_SIZE][MAXDIM]) {
    __attribute__((xcl_pipeline_loop(1)))
    for (int j = 0; j < n_dim; j++) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            res_out[i][j] = res_in[j][i];
        }
    }
}

__kernel
__attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__ ((xcl_dataflow)) 
void events_kernel(__global const double *divisions_ext,
        const int n_dim, const double xjac, const int index,
        __global double all_res[BUFFER_SIZE], __global int div_indexes[BUFFER_SIZE][MAXDIM]) {
    // Copy in the external input
    // this external input can safely enter through HBM
    double divisions[MAXDIM][BINS_MAX];
    read_divs(divisions_ext, divisions, n_dim);

    // Now let us create some intermediate arrays
    double randoms[MAXDIM][BUFFER_SIZE];  // we want to partition this!
    double integrator_input[BUFFER_SIZE][MAXDIM];
    int indexes[MAXDIM][BUFFER_SIZE];
    double wgts[BUFFER_SIZE];
    double wgt_buffer[BUFFER_SIZE];

    double all_results[BUFFER_SIZE];
    STATE_RNG rng[BUFFER_SIZE];
    // Now initialize the random number generator
    rng_initializer(index, rng);

    // Compute all the randoms
    for (int j = 0; j < MAXDIM; j++) {
        // The first step is to compute all random numbers
        if (j == 0) {
            initialize_1d(wgt_buffer, xjac);
        }
        random_computer(divisions[j], wgt_buffer, rng, randoms[j], indexes[j], wgts);
        if (j == n_dim-1) break;
        copybuffer_1d(wgts, wgt_buffer);
    }
    // Secondly we need to digest all random data
    digest_random(randoms, n_dim, integrator_input);

    // Compute now all the results
    integral_computer(integrator_input, wgts, n_dim, all_results);

////    // And now burst write the results back to the host
    return_2d(indexes, n_dim, div_indexes);
    return_1d(all_results, all_res);
}

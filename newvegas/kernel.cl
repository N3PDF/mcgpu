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
    for (int j = 0; j < n_dim; j++) {
        const int jdx = j*BINS_MAX;
        for (int k = 0; k < BINS_MAX; k++) {
            div_out[j][k] = div_in[jdx + k];
        }
    }
}

void random_initializer(const int index, double rng[BUFFER_SIZE]) {
    for (int i = 0; i < ndim; i++) {
        const idx = index+i;
        random_initializer(rng[i], idx, idx);
    }
}

#define reg_i 0.0
#define reg_f 1.0
void random_computer(const double divisions[MAXDIM][BINS_MAX], const STATE_RNG rngs[BUFFER_SIZE],
        const int n_dim,
        double randoms[BUFFER_SIZE][MAX_DIM], int indexes[BUFFER_SIZE][MAX_DIM], double wgts[BUFFER_SIZE]) {
    // hey, I just found you
    // and this is craaaazy
    // but here's my arrays
    // compile me maybe
    for (int j = 0; j < n_dim; j++) {
        const jdx = j*BINS_MAX;
        for (int i = 0; i < BUFFER_SIZE; i++) {
            const double rn = GEN_RAN(rng[i]);
            const double xn = BINS_MAX*(1.0 - rn);
            int int_xn = (int) max(0, min( (int) xn, BINS_MAX));
            const int idx = jdx + int_xn;
            const double aux_rand = xn - int_xn;
            if (int_xn > 0) {
                x_ini = divisions[idx - 1];
            } else {
                x_ini = 0.0;
            }
            const double xdelta = divisions[idx] - x_ini;
            const double rand_x = x_ini + xdelta*aux_rand;
            if (j == 0) {
                wgts[i] = xdelta*BINS_MAX;
            } else {
                wgts[i] *= xdelta*BINS_MAX;
            }
            randoms[i][j] = reg_i + rand_x*(reg_f-reg_i);
            indexes[i][j] = int_xn;
        }
    }
}

void integral_computer(const double randoms[BUFFER_SIZE][MAXDIM], 
        const int n_dim,
        double all_results[BUFFER_SIZE]) {
    for (int i = 0; i < BUFFER_SIZE; i++) {
        all_results[i] = integrand(n_dim, &randoms[i]);
    }
}

void return_1d(const double res_in[BUFFER_SIZE], __global double res_out[BUFFER_SIZE]) {
    for (int i = 0; i < BUFFER_SIZE; i++) {
        res_out[i] = res_in[i];
    }
}
void return_2d(const double res_in[BUFFER_SIZE], const int n_dim, __global double res_out[BUFFER_SIZE]) {
    for (int j = 0; j < n_dim; j++) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            res_out[i][j] = res_in[i][j];
        }
    }
}


__kernel
void events_kernel(__global const double *divisions_ext,
        const int n_dim, const double xjac, const int index,
        __global const double all_res[BUFFER_SIZE], __global const double *div_indexes) {

    // Copy in the external input
    // this external input can safely enter through HBM
    double divisions[MAXDIM][BINS_MAX];
    read_divs(divisions_ext, divisions, n_dim);

    // Now let us create some intermediate arrays
    double randoms[BUFFER_SIZE][MAX_DIM];  // we want to partition this!
    double wgts[BUFFER_SIZE];
    int indexes[BUFFER_SIZE][MAX_DIM];
    double all_results[BUFFER_SIZE];
    STATE_RNG rng[BUFFER_SIZE];

    // Now initialize the random number generator
    random_initializer(index, rng);

    // Compute all the randoms
    random_computer(divisions, rng, n_dim, randoms, indexes, wgts);

    // Compute now all the results
    integral_computer(randoms, n_dim, all_results);

    // And now burst write the results back to the host
    return_1d(all_results, all_res);
    return_2d(indexex, n_dim, div_indexes);

}

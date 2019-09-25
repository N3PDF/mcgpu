#include <curand.h>
#include <curand_kernel.h>

#define BINS_MAX 30
#define ALPHA 0.1

extern "C" {
__global__
void events_kernel(double *all_randoms, double *all_xwgts, int n, int n_events, double xjac, double *all_res, double *all_res2) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int block_size = blockDim.x;

    int index = block_id*block_size + thread_id;
    int grid_dim = gridDim.x;
    int stride = block_size * grid_dim;

    // shared lepage
    double a = 0.1;
    double pref = pow(1.0/a/sqrt(M_PI), n);

    for (int i = index; i < n_events; i += stride) {
        double wgt = all_xwgts[i]*xjac;

        double coef = 0.0;
        for (int j = 0; j < n; j++) {
            coef += pow( (all_randoms[i*n + j] - 1.0/2.0)/a, 2 );
        }
        double lepage = pref*exp(-coef);

        double tmp = wgt*lepage;
        all_res[i] = tmp;
        all_res2[i] = pow(tmp,2);
    }
}

__global__
void generate_random_array_kernel(const int n_events, const int n_dim, const double *divisions, double *all_randoms, double *all_wgts, int *all_div_indexes) {
    double reg_i = 0.0;
    double reg_f = 1.0;

    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int block_size = blockDim.x;

    int index = block_id*block_size + thread_id;
    int grid_dim = gridDim.x;
    int stride = block_size * grid_dim;

    // Use curandState_t to keep track of the seed, which is thread dependent
    curandState_t state;
    // seed-sequence-offset
    curand_init(index, 0, 0, &state);

    for (int j = index; j < n_events; j+= stride) {
        double wgt = 1.0;
        for (int i = 0; i < n_dim; i++) {
            double rn = curand_uniform_double(&state);
            double xn = BINS_MAX*(1.0 - rn);
            int int_xn = max(0, min( (int) xn, BINS_MAX));
            double aux_rand = xn - int_xn;
            double x_ini = 0.0;
            if (int_xn > 0) {
                x_ini = divisions[BINS_MAX*i + int_xn-1];
            }
            double xdelta = divisions[BINS_MAX*i + int_xn] - x_ini;
            double rand_x = x_ini + xdelta*aux_rand;
            wgt *= xdelta*BINS_MAX;
            // Now we need to add an offset to the arrays
            all_randoms[j*n_dim + i] = reg_i + rand_x*(reg_f - reg_i);
            all_div_indexes[j*n_dim + i] = int_xn;
            }
        all_wgts[j] = wgt;
    }
}

}

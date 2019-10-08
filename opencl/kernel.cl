#include "definitions.h"

// Include opencl files
#include "random.cl"
#include "integrand.cl"

double generate_random_array(STATE_RNG *rng, const int n_dim, __global const double *divisions, double *randoms, int *div_indexes) {
    const double reg_i = 0.0;
    const double reg_f = 1.0;
    double wgt = 1.0;
    for (int j = 0; j < n_dim; j++) {
        const double rn = GEN_RAN(rng);
        const double xn = BINS_MAX*(1.0 - rn);
        int int_xn = max(0, min( (int) xn, BINS_MAX));
        const double aux_rand = xn - int_xn;
        double x_ini = 0.0;
        if (int_xn > 0) {
            x_ini = divisions[BINS_MAX*j + int_xn-1];
        }
        const double xdelta = divisions[BINS_MAX*j + int_xn] - x_ini;
        const double rand_x = x_ini + xdelta*aux_rand;
        wgt *= xdelta*BINS_MAX;
        randoms[j] = reg_i + rand_x*(reg_f - reg_i);
        div_indexes[j] = int_xn;
        }
    return wgt;
}

// Kernel to be run per event
__attribute__((vec_type_hint(double)))
__attribute__((xcl_zero_global_work_offset))
__attribute__((max_work_group_size(MAXTHREADS, 1, 1)))
__kernel void events_kernel(__global const double *divisions, const int n_dim, const int events_per_kernel, const double xjac, __global double *all_res, __global double *arr_res2) {
    const int block_id = get_group_id(0);
    const int thread_id = get_local_id(0);
    const int block_size = get_local_size(0);

    const int index = block_id*block_size + thread_id;
    const int grid_dim = get_num_groups(0);
    const int stride = block_size * grid_dim;
    double randoms[MAXDIM];
    int indexes[MAXDIM];
    int seed = index;

    STATE_RNG rng;
    RGN_INITIALIZER(&rng, 0, index);

    const int idx_res2 = index*BINS_MAX*n_dim;
    // tmp variables
    double final_result = 0.0;
    double tmp_arr_res2[BINS_MAX*MAXDIM];

    for (int i = 0; i < n_dim*BINS_MAX; i++) {
        tmp_arr_res2[i] = 0.0;
    }
//__attribute__((opencl_unroll_hint(1)))
    for (int i = 0; i < events_per_kernel; i++) {
//        printf("%d\n", i);
        const double wgt = generate_random_array(&rng, n_dim, divisions, &randoms, &indexes);
        const double lepage = integrand(n_dim, &randoms);
        const double tmp = xjac*wgt*lepage;
        final_result += tmp;
        for (int j = 0; j < n_dim; j++) {
            const int idx = indexes[j]*n_dim + j;
            tmp_arr_res2[idx] += pow(tmp,2);
        }
    }
    all_res[index] = final_result;
    // I find this unnecessary...
    for (int i = 0; i < n_dim*BINS_MAX; i++) {
        arr_res2[idx_res2 + i] = tmp_arr_res2[i];
    }
}

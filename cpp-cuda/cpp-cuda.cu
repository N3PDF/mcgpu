#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <sys/param.h>
#include <sys/time.h>

// Cuda includes
#include <curand.h>
#include <curand_kernel.h>

#define BINS_MAX 30
#define ALPHA 0.1

double gpu_double_reduction(const int array_size, const double *target_array) {
    // Make sense only for arrays which are in the GPU already
    double res = 0;
    for (int i = 0; i < array_size; i++) {
        res += target_array[i];
    }
    return res;
}

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
            double rn = (double) curand(&state)/RAND_MAX/2; //unsigned?
            double xn = BINS_MAX*(1.0 - rn);
            int int_xn = MAX(0, MIN( (int) xn, BINS_MAX));
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



void rebin(const double *rw, const double rc, double *subdivisions) {
    int k = -1;
    double dr = 0.0;
    double aux[BINS_MAX];
    for (int i = 0; i < BINS_MAX-1; i++){
        double old_xi = 0.0;
        while (rc > dr) {
            k += 1;
            dr += rw[k];
        }
        if (k > 0) {
            old_xi = subdivisions[k-1];
        }
        double old_xf = subdivisions[k];
        dr -= rc;
        double delta_x = old_xf - old_xi;
        aux[i] = old_xf - delta_x*(dr / rw[k]);
    }
    aux[BINS_MAX-1] = 1.0;
    for (int i = 0; i < BINS_MAX; i ++) {
        subdivisions[i] = aux[i];
    }
}

void refine_grid(const double *res_sq, double *subdivisions){
    double aux[BINS_MAX];
    double tmp = (res_sq[0] + res_sq[1])/2.0;
    double aux_sum = tmp;
    aux[0] = tmp;
    for (int i = 1; i < BINS_MAX-1; i++) {
        tmp = MAX( (res_sq[i-1]+res_sq[i]+res_sq[i+1])/3.0, 1e-30 );
        aux_sum += tmp;
        aux[i] = tmp;
    }
    tmp = (res_sq[BINS_MAX-2] + res_sq[BINS_MAX-1])/2.0;
    aux_sum += tmp;
    aux[BINS_MAX-1] = tmp;
    double rw[BINS_MAX];
    double rc = 0.0;
    for (int i = 0; i < BINS_MAX-1; i ++) {
        tmp = pow( (1.0 - aux[i]/aux_sum)/(log(aux_sum) - log(aux[i])), ALPHA );
        rw[i] = tmp;
        rc += tmp;
    }
    rc = rc/BINS_MAX;
    rebin(rw, rc, subdivisions);
}



int vegas(const int warmup, const int n_dim, const int n_iter, const int n_events, double *final_result, double *sigma) {
    srand(0);
    double xjac = 1.0/n_events;
    double *divisions;
    // Allocate divisions in unified memory
    cudaMallocManaged(&divisions, n_dim*BINS_MAX*sizeof(double));
    for( int j = 0; j < n_dim; j++ ){
        divisions[j*BINS_MAX] = 1.0;
    }
    // fake initialization at the beginning
    double rw_tmp[BINS_MAX] = { 1.0 };
    for( int j = 0; j < n_dim; j++ ){
        rebin(rw_tmp, 1.0/BINS_MAX, &divisions[j*BINS_MAX]);
    }

    double total_res = 0.0;
    double total_weight = 0.0;

    // Both threadas and blocks map to the total number of events
    int threads = 256;
    int blocks = (n_events + threads - 1)/threads;

    for( int k = 0; k < n_iter; k++ ) {
        double res = 0.;
        double res2 = 0.;
        double sigma = 0.;

        // Array with the results by bin
        double *arr_res2;
        arr_res2 = (double *) calloc(n_dim*BINS_MAX, sizeof(double));

        // input arrays
        double *all_randoms, *all_xwgts;
        int NN = n_dim*n_events;
        int *all_div_indexes;
        cudaMalloc(&all_randoms, NN*sizeof(double));
        cudaMalloc(&all_xwgts, n_events*sizeof(double));
        cudaMallocManaged(&all_div_indexes, NN*sizeof(int));

        // output arrays
        double *all_res, *all_res2;
        cudaMallocManaged(&all_res, n_events*sizeof(double));
        cudaMallocManaged(&all_res2, n_events*sizeof(double));

        // Before going in we might have change divisions so force a sync in
        cudaDeviceSynchronize();
        // Generate the random numbers on the GPU
        generate_random_array_kernel<<<blocks, threads>>>(n_events, n_dim, divisions, all_randoms, all_xwgts, all_div_indexes);
        // Use them to generate results
        events_kernel<<<blocks, threads>>>(all_randoms, all_xwgts, n_dim, n_events, xjac, all_res, all_res2);

        // Synchronize memory
        cudaDeviceSynchronize();

        // Poor's man reduction
        res = gpu_double_reduction(n_events, all_res);
        res2 = gpu_double_reduction(n_events, all_res2);

        for (int i = 0; i < n_events; i++) {
            for (int j = 0; j < n_dim; j++) {
                arr_res2[j*BINS_MAX + all_div_indexes[i*n_dim + j]] += all_res2[i];
            }
        }

        cudaFree(all_res);
        cudaFree(all_res2);
        cudaFree(all_xwgts);
        cudaFree(all_randoms);
        cudaFree(all_div_indexes);

        double err_tmp2 = MAX((n_events*res2 - res*res)/(n_events-1.0), 1e-30);
        sigma = sqrt(err_tmp2);
        printf("For iteration %d, result: %1.5f +- %1.5f\n", k+1, res, sigma);

        for (int j = 0; j < n_dim; j ++) {
            refine_grid(&arr_res2[j*BINS_MAX], &divisions[j*BINS_MAX]);
        }

        double wgt_tmp = 1.0/pow(sigma, 2);
        total_res += res*wgt_tmp;
        total_weight += wgt_tmp;

        free(arr_res2);
    }

    
    *final_result = total_res/total_weight;
    *sigma = sqrt(1.0/total_weight);

    cudaFree(divisions);

    return 0;
}

int main(int argc, char *argv[]) {
    int n_dim = 7;
    int n_events = (int) 1e6;
    int n_iter = 5;

    double res, sigma;
    struct timeval start, stop;

    gettimeofday(&start, 0);
    int state = vegas(1, n_dim, n_iter, n_events, &res, &sigma);
    gettimeofday(&stop, 0);

    if (state != 0) {
       printf("Something went wrong\n");
    }

    printf("Final result: %1.5f +- %1.5f\n", res, sigma);
    double result = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*1e-6;
    printf("It took: %1.7f seconds\n", result);

    return 0;
}

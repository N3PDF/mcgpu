#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/param.h>
#include <omp.h>
#include "Vegas.h"

double internal_rand(){
    double x = (double) rand()/RAND_MAX;
    return x;
}

double generate_random_array(const int n_dim, const double *divisions, double *x, int *div_index) {
    double reg_i = 0.0;
    double reg_f = 1.0;
    double wgt = 1.0;
    for (int i = 0; i < n_dim; i++) {
        double rn = internal_rand();
        double xn = BINS_MAX*(1.0 - rn);
        int int_xn = MAX(0, MIN( (int) xn, BINS_MAX));
        double aux_rand = xn - int_xn;
        double x_ini = 0.0;
        if (int_xn > 0) {
            x_ini = divisions[BINS_MAX*i + int_xn-1];
        }
        double xdelta = divisions[BINS_MAX*i + int_xn] - x_ini;
        double rand_x = x_ini + xdelta*aux_rand;
        x[i] = reg_i + rand_x*(reg_f - reg_i);
        wgt *= xdelta*BINS_MAX;
        div_index[i] = int_xn;
    }
    return wgt;
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

int vegas(double (*f_integrand)(double*, int), const int warmup, const int n_dim, const int n_iter, const int n_events, double *final_result, double *sigma) {
    srand(0);
    double xjac = 1.0/n_events;
    double *divisions;
    divisions = (double *) calloc(n_dim*BINS_MAX, sizeof(double));
    for( int j = 0; j < n_dim; j++ ){
        divisions[j*BINS_MAX] = 1.0;
    }
    // fake initialization at the beginning
    double rw_tmp[BINS_MAX] = { [0 ... BINS_MAX-1] = 1.0 };
    for( int j = 0; j < n_dim; j++ ){
        rebin(rw_tmp, 1.0/BINS_MAX, &divisions[j*BINS_MAX]);
    }

    double total_res = 0.0;
    double total_weight = 0.0;

    #if defined _OPENMP
    printf("OMP active\n");
    printf("Maximum number of threads: %d\n", omp_get_num_procs() );
    printf("Number of threads selected: %d\n", omp_get_max_threads() );
    #endif

    for( int k = 0; k < n_iter; k++ ) {
        double res = 0.;
        double res2 = 0.;
        double sigma = 0.;

        double *arr_res2;
        arr_res2 = (double *) calloc(n_dim*BINS_MAX, sizeof(double));
        int *div_index;
        double *x;

        #pragma omp parallel private(x, div_index)
        {
            div_index = (int *) malloc(n_dim*sizeof(int));
            x = (double *) malloc(n_dim*sizeof(double));
            #pragma omp for reduction( + : res, res2 )
            for ( int i = 0;  i < n_events; i++ ) {
                double xwgt;
                #pragma omp critical 
                {
                    xwgt = generate_random_array(n_dim, divisions, x, div_index);
                }
                double wgt = xwgt*xjac;
                double tmp = wgt*f_integrand(x, n_dim);
                double tmp2 = pow(tmp,2); 
                res += tmp;
                res2 += tmp2;

                #pragma omp critical 
                { // array reduction must be done manually in C?
                    for( int j = 0; j < n_dim; j++ ){
                        arr_res2[j*BINS_MAX + div_index[j]] += tmp2;
                    }
                }

            }
            free(x);
            free(div_index);
        }
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

    free(divisions);

    return 0;
}


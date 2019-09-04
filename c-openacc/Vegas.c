#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/param.h>
#include "Vegas.h"
#include "integrands/lepage_test.h"


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

int vegas(const int warmup, const int n_dim, const int n_iter, const int n_events, double *final_result, double *sigma) {
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

    for( int k = 0; k < n_iter; k++ ) {
        double res = 0.;
        double res2 = 0.;
        double sigma = 0.;

        double *arr_res2;
        arr_res2 = (double *) calloc(n_dim*BINS_MAX, sizeof(double));

        double *all_randoms, *all_xwgts;
        int *all_div_indexes;
        int NN = n_dim*n_events;
        all_randoms = (double *) malloc(sizeof(double)*NN);
        all_xwgts = (double *) malloc(sizeof(double)*n_events);
        all_div_indexes = (int *) malloc(sizeof(int)*NN);
        for (int i = 0; i < n_events; i++) {
            all_xwgts[i] = generate_random_array(n_dim, divisions, &all_randoms[i*n_dim], &all_div_indexes[i*n_dim]);
        }

        double *all_gpu_results;
        all_gpu_results = (double *) malloc(sizeof(double)*n_events);

        #pragma acc data copyin(all_xwgts[0:n_events], all_randoms[0:NN])
        #pragma acc data copyout(all_gpu_results[0:n_events])
        #pragma acc parallel loop reduction(+:res, res2)
        for (int i = 0; i < n_events; i++) {
            double wgt = all_xwgts[i]*xjac;

            // lepage_test
            double a = 0.1;
            double coef = 0.0;
            double pref = pow(1.0/a/sqrt(M_PI), n_dim);
            for (int j = 0; j < n_dim; j++) {
                    coef += pow((all_randoms[i*n_dim + j] - 1.0/2.0)/a, 2);
                }
            double lepage = pref*exp(-coef);

            double tmp = wgt*lepage;
            double tmp2 = pow(tmp, 2);
            res += tmp;
            res2 += tmp2;

            all_gpu_results[i] = tmp2;

        }


        for (int i = 0; i < n_events; i++) {
            for (int j = 0; j < n_dim; j++) {
                arr_res2[j*BINS_MAX + all_div_indexes[i*n_dim + j]] += all_gpu_results[i];
            }
        }
        
        free(all_gpu_results);
        free(all_xwgts);
        free(all_randoms);
        free(all_div_indexes);

        // temporary
//        double *all_randoms, *all_xwgts;
//        int *all_div_indexes;
//        #pragma acc declare create(all_randoms, all_xwgts, all_div_indexes)
//        all_randoms = (double *) malloc(sizeof(double)*n_dim*n_events);
//        all_xwgts = (double *) malloc(n_events*sizeof(double));
//        all_div_indexes = (int *) malloc(sizeof(int)*n_dim*n_events);
//        printf("Size: %lu\n", sizeof(int)*n_dim*n_events);
//        for (int i = 0; i < n_events; i++) {
//            all_xwgts[i] = generate_random_array(n_dim, divisions, &all_randoms[i*n_dim], &all_div_indexes[i*n_dim]);
//        }
//        int N = n_dim*n_events;

//        #pragma acc data copyin(all_div_indexes[0:N], all_xwgts[0:n_events], all_randoms[0:N])
//        #pragma acc kernels 
//        for ( int i = 0;  i < n_events; i++ ) {
//            double wgt = all_xwgts[i]*xjac;
//            double tmp = wgt;//*lepage_test(&all_randoms[i*n_dim], n_dim);
//            double tmp2 = pow(tmp,2); 
//            res += tmp;
//            res2 += tmp2;
//
//            for( int j = 0; j < n_dim; j++ ){
//                arr_res2[j*BINS_MAX + all_div_indexes[i*n_dim + j]] += tmp2;
//            }
//        }

//        free(all_randoms);
//        free(all_xwgts);
//        free(all_div_indexes);

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


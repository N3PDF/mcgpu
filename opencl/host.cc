#include <vector>
#include <string>
#include <iostream>
#include <math.h>
#include <sys/time.h>

#include "mccl.h"
#include <CL/cl2.hpp>

using namespace std;


#define BINS_MAX 50
#define ALPHA 1.5

double internal_rand(){
    double x = (double) rand()/RAND_MAX;
    return x;
}

double* generate_random_array(const int n_events, const int n_dim, const double *divisions, double *x, int *div_index) {

    double *all_wgts = (double*) aligned_alloc(n_events, n_events * sizeof(double));;
    for (int j = 0; j < n_events; j++)
    {
        double reg_i = 0.0;
        double reg_f = 1.0;
        double wgt = 1.0;
        int index = j*n_dim;
        for (int i = 0; i < n_dim; i++) {
            double rn = internal_rand();
            double xn = BINS_MAX*(1.0 - rn);
            int int_xn = max(0, min( (int) xn, BINS_MAX));
            double aux_rand = xn - int_xn;
            double x_ini = 0.0;
            if (int_xn > 0) {
                x_ini = divisions[BINS_MAX*i + int_xn-1];
            }
            double xdelta = divisions[BINS_MAX*i + int_xn] - x_ini;
            double rand_x = x_ini + xdelta*aux_rand;
            x[i+index] = reg_i + rand_x*(reg_f - reg_i);
            wgt *= xdelta*BINS_MAX;
            div_index[i+index] = int_xn;
        }
        all_wgts[j] = wgt;
    }
    return all_wgts;
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
        tmp = max( (res_sq[i-1]+res_sq[i]+res_sq[i+1])/3.0, 1e-30 );
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

int vegas(std::string kernel_file, const int device_idx, const int warmup, const int n_dim, const int n_iter, const int n_events, double *final_result, double *sigma) {
    // Auxiliary variables
    int NN = n_dim*n_events;
    int err;

    // OpenCL initialization

    // Read the device 
    auto device = get_default_device(device_idx); // platform 0 == GPU

    // Create context, queue and program
    cl::Context context( {device} );
    cl::CommandQueue q(context, device);
    cl::Program program;
    if (device_idx == 1) {
        program = read_program_from_bin(kernel_file, context, device);
    } else {
        program = read_program_from_file(kernel_file, context, device);
    }

    // Red the kernel out of the program
    cl::Kernel kernel(program, "events_kernel", &err);

    // Now allocate the memory buffers on the device
    cl::Buffer buffer_all_randoms(context, CL_MEM_READ_ONLY, sizeof(double) * NN, NULL, &err);
    cl::Buffer buffer_all_xwgts(context, CL_MEM_READ_ONLY, sizeof(double) * n_events, NULL, &err);
    cl::Buffer buffer_all_res(context, CL_MEM_WRITE_ONLY, sizeof(double) * n_events, NULL, &err);
    cl::Buffer buffer_all_res2(context, CL_MEM_WRITE_ONLY, sizeof(double) * n_events, NULL, &err);
    // end OCL initialization

    srand(0);
    double xjac = 1.0/n_events;
    double *divisions = new double[n_dim*BINS_MAX];
    for( int j = 0; j < n_dim; j++ ){
        divisions[j*BINS_MAX] = 1.0;
    }
    // fake initialization at the beginning
    double rw_tmp[BINS_MAX];
    std::fill_n(rw_tmp, BINS_MAX, 1.0);
    for( int j = 0; j < n_dim; j++ ){
        rebin(rw_tmp, 1.0/BINS_MAX, &divisions[j*BINS_MAX]);
    }

    double total_res = 0.0;
    double total_weight = 0.0;

    for( int k = 0; k < n_iter; k++ ) {
        cout << "Starting iteration " << k << endl;
        double res = 0.;
        double res2 = 0.;
        double sigma = 0.;

        //double *all_randoms = new double[NN];
        double *all_randoms = (double*) aligned_alloc(NN, NN * sizeof(double));
        int *all_div_indexes = (int*) aligned_alloc(NN, NN * sizeof(int));
        for (int i = 0; i < NN; i++) all_div_indexes[i] = 0;

        // output arrays
        double *all_res = (double*) aligned_alloc(n_events, n_events * sizeof(double));
        double *all_res2 = (double*) aligned_alloc(n_events, n_events * sizeof(double));

        double *all_xwgts = generate_random_array(n_events, n_dim, divisions, all_randoms, all_div_indexes);

        cl::Event wb_event1, wb_event2;
        q.enqueueWriteBuffer(buffer_all_randoms, CL_FALSE, 0, sizeof(double)*NN, all_randoms, NULL, &wb_event1);
        q.enqueueWriteBuffer(buffer_all_xwgts, CL_FALSE, 0, sizeof(double)*n_events, all_xwgts, NULL, &wb_event2);
        vector<cl::Event> wb_events = {wb_event1, wb_event2};

        cl_uint narg = 0;
        err = kernel.setArg(narg++, buffer_all_randoms);
        err = kernel.setArg(narg++, buffer_all_xwgts);
        err = kernel.setArg(narg++, n_dim);
        err = kernel.setArg(narg++, n_events);
        err = kernel.setArg(narg++, xjac);
        err = kernel.setArg(narg++, buffer_all_res);
        err = kernel.setArg(narg++, buffer_all_res2);

        // Launch the Kernel
        cl::Event kernel_event;
        err = q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n_events), cl::NullRange, &wb_events, &kernel_event);
        vector<cl::Event> wait_what = {kernel_event}; // I am sure there is a less stupid way of doing this...

        // Copy the result from the device
        err = q.enqueueReadBuffer(buffer_all_res, CL_TRUE, 0, sizeof(double)*n_events, all_res, &wait_what, NULL);
        err = q.enqueueReadBuffer(buffer_all_res2, CL_TRUE, 0, sizeof(double)*n_events, all_res2, &wait_what, NULL);

        q.finish();

        for (int i = 0; i < n_events; i++)
        {
            res += all_res[i];
            res2 += all_res2[i];
        }

        double *arr_res2 = new double[n_dim*BINS_MAX]();
        for( int i = 0; i < n_events; i++ )
            for (int j = 0; j < n_dim; j++)
                arr_res2[j*BINS_MAX + all_div_indexes[i*n_dim + j]] += all_res2[i];

        double err_tmp2 = max((n_events*res2 - res*res)/(n_events-1.0), 1e-30);
        sigma = sqrt(err_tmp2);
        printf("For iteration %d, result: %1.5f +- %1.5f\n", k+1, res, sigma);

        for (int j = 0; j < n_dim; j ++)
            refine_grid(&arr_res2[j*BINS_MAX], &divisions[j*BINS_MAX]);

        double wgt_tmp = 1.0/pow(sigma, 2);
        total_res += res*wgt_tmp;
        total_weight += wgt_tmp;

        free(all_randoms);
        free(all_div_indexes);
        free(all_xwgts);
        free(all_res);
        free(all_res2);

        delete[] arr_res2;
    }


    *final_result = total_res/total_weight;
    *sigma = sqrt(1.0/total_weight);

    delete[] divisions;

    return 0;
}

int main(int argc, char **argv) {

    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage %s number_of_events number_of_dimensions kernel_file device\n", argv[0]);
        exit(0);
    }
    string kernel_file = "kernel.cl";
    if (argc > 3) {
        // Careful, this will run even if the file does not exist
        kernel_file = argv[3];
    }
    int device_idx = 0;
    if (argc > 4) {
        device_idx = atoi(argv[4]);
    }
    int n_events = atoi(argv[1]);
    int n_dim = atoi(argv[2]);
    int n_iter = 5;

    double res, sigma;
    struct timeval start, stop;

    gettimeofday(&start, 0);
    int state = vegas(kernel_file, device_idx, 1, n_dim, n_iter, n_events, &res, &sigma);
    gettimeofday(&stop, 0);

    if (state != 0) {
       printf("Something went wrong\n");
    }

    printf("Final result: %1.5f +- %1.5f\n", res, sigma);
    double result = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*1e-6;
    printf("It took: %1.7f seconds\n", result);

    return 0;
}

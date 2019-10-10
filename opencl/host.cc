#include <vector>
#include <string>
#include <iostream>
#include <math.h>
#include <sys/time.h>

#include "mccl.h"
#include "definitions.h"
#include <CL/cl2.hpp>

using namespace std;
template <typename T>
using aligned_vector = vector<T, aligned_allocator<T>>;

void rebin(const vector<double> rw, const double rc, double *subdivisions) {
    int k = -1;
    double dr = 0.0;
    vector<double> aux(BINS_MAX);
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
    vector<double> aux(BINS_MAX);
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
    vector<double> rw(BINS_MAX);
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

    // At this point we have all necessary information to decide how many kernels to launch and how 
    // many events should each kernel do
    // it is stupid to send one thread per event, but it is a good idea to go beyond the maximum number of threads here
    const int max_device_threads = min(MAXTHREADS, n_events); // max number of parallel kernels to be launched 
    const int events_per_kernel = (int) max(1, n_events/max_device_threads); // TODO: ensure the total number is n_events at the end
    cout << "Threads to be sent: " << max_device_threads << ", events per kernel: " << events_per_kernel << endl;

    // Set the sizes of all the arrays of this run
    // Auxiliary variables
    const int div_size = n_dim*BINS_MAX;
    const int arr_size = max_device_threads*BINS_MAX*n_dim;
    const int all_res_size = max_device_threads;

    // Declare HOST arrays
    aligned_vector<double> divisions(div_size);
    aligned_vector<double> all_res(all_res_size);
    aligned_vector<double> arr_res2(arr_size);

    // Initialize integration
    srand(0);
    const double xjac = 1.0/n_events;
    for(int j = 0; j < n_dim; j++ ){
        divisions[j*BINS_MAX] = 1.0;
    }
    // fake initialization at the beginning
    vector<double> rw_tmp(BINS_MAX, 1.0);
    for( int j = 0; j < n_dim; j++ ){
        rebin(rw_tmp, 1.0/BINS_MAX, &divisions[j*BINS_MAX]);
    }
    double total_res = 0.0;
    double total_weight = 0.0;



    // OpenCL initialization
    int err;

    // Read the device 
    const auto device = get_default_device(device_idx); // platform 0 == GPU

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
    // Copy-IN buffer
    cl::Buffer buffer_divisions(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double) * div_size, divisions.data(), &err);
    // Copy-OUT buffers
    cl::Buffer buffer_arr_res2(context, CL_MEM_USE_HOST_PTR |  CL_MEM_WRITE_ONLY, sizeof(double) * arr_size, arr_res2.data(), &err);
    cl::Buffer buffer_all_res(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(double)*all_res_size, all_res.data(), &err);
    // end OCL initialization
    if (err) {
        cout << "Some error while allocating buffers in the device" << endl;
        return -1;
    }

    // Prepare the event kernel
    cl_uint narg = 0;
    err = kernel.setArg(narg++, buffer_divisions);
    err = kernel.setArg(narg++, n_dim);
    err = kernel.setArg(narg++, events_per_kernel);
    err = kernel.setArg(narg++, xjac);
    err = kernel.setArg(narg++, buffer_all_res);
    err = kernel.setArg(narg++, buffer_arr_res2);
    if (err) {
        cout << "Some error while setting the kernel arguments" << endl;
        return -1;
    }

    for( int k = 0; k < n_iter; k++ ) {
        cout << "Starting iteration " << k << endl;

        vector<cl::Event> all_events(1);
        // Launch the kernel
        err = q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(max_device_threads), cl::NullRange, NULL, &all_events[0]);
        // Copy the result from the device
        err = q.enqueueMigrateMemObjects({buffer_all_res, buffer_arr_res2}, CL_MIGRATE_MEM_OBJECT_HOST, &all_events); 
        q.finish();

        double res = 0.;
        double res2 = 0.;
        for (int i = 0; i < max_device_threads; i++) {
            res += all_res[i];
            const int idx_t = i*BINS_MAX*n_dim;
            for (int j = 0; j < BINS_MAX; j++) {
                const int idx_b = idx_t + j*n_dim;
                res2 += arr_res2[idx_b];
            }
        }

        const double err_tmp2 = max((n_events*res2 - res*res)/(n_events-1.0), 1e-30);
        const double sigma_tmp = sqrt(err_tmp2);
        printf("For iteration %d, result: %1.5f +- %1.5f\n", k+1, res, sigma_tmp);

        const double wgt_tmp = 1.0/pow(sigma_tmp, 2);
        total_res += res*wgt_tmp;
        total_weight += wgt_tmp;

        for (int j = 0; j < n_dim; j ++)
            refine_grid(&arr_res2[j*BINS_MAX], &divisions[j*BINS_MAX]);

        if ( k != n_iter ) {
            // If we are still doing the integration, rewrite the divisions buffer
            // OPENCL will ensure this is a blocking call so no need to worry about saving events or whatever
            q.enqueueWriteBuffer(buffer_divisions, CL_TRUE, 0, sizeof(double)*div_size, divisions.data());
        }
    }

    *final_result = total_res/total_weight;
    *sigma = sqrt(1.0/total_weight);

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

    if (n_dim > MAXDIM) {
        cout << "ERROR: " << n_dim << " over the maximum number of dimensions allowed: " << MAXDIM << endl;
        return -1;
    }

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

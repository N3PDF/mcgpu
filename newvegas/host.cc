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

int vegas(std::string kernel_file, const int device_idx, const int warmup, const int n_dim, const int n_iter, const int n_events, double *vegas_result, double *sigma) {

    // Set the sizes of all the arrays of this run
    // Auxiliary variables
    const int div_size = n_dim*BINS_MAX;

    // Declare HOST arrays
    aligned_vector<double> divisions(div_size);

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

    // First decide how many concurrent calculations we want
    const int n_concurrent = 1;

    string kernel_name = "events_kernel";

    // declare complicated host arrays
//    vector<aligned_vector<double>> results(
//            n_concurrent, aligned_vector<double>(BUFFER_SIZE)
//            );
//    vector<aligned_vector<int>> indexes(
//            n_concurrent, aligned_vector<int>(BUFFER_SIZE*n_dim)
//            );

    vector<aligned_vector<double>> results(n_concurrent);
    vector<aligned_vector<int>> indexes(n_concurrent);
    for (int i = 0; i < n_concurrent; i++){
        results[i] = aligned_vector<double>(BUFFER_SIZE, 0.0);
        indexes[i] = aligned_vector<int>(BUFFER_SIZE*MAXDIM, 0);
    }

    // Now let us start by declaring everything that will be reused
    vector<cl::Kernel> all_kernels(n_concurrent);

    vector<cl::Buffer> b_divisions(n_concurrent);
    vector<cl::Buffer> b_all_res(n_concurrent), b_div_indexes(n_concurrent);

    for (int i = 0; i < n_concurrent; i++) {
        string cuname = kernel_name ;//+ ":{" + kernel_name + "_" + to_string(i+1) + "}";
        all_kernels[i] = cl::Kernel(program, cuname.c_str(), &err);
        if(err) { 
            cout << " > ERROR reading kernel " << cuname << endl;
            return -1;
        }
        b_divisions[i] = cl::Buffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double) * div_size, divisions.data(), &err);
        b_all_res[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(double)*BUFFER_SIZE, results[i].data(), &err);
        b_div_indexes[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(int)*BUFFER_SIZE*MAXDIM, indexes[i].data(), &err);
        if(err) {
            cout << "Error preparing the buffers" << endl;
            return -1;
        }


        // Now set the arguments for the kernel
        cl_uint narg = 0;
        err = all_kernels[i].setArg(narg++, b_divisions[i]);
        err = all_kernels[i].setArg(narg++, n_dim);
        err = all_kernels[i].setArg(narg++, xjac);
        err = all_kernels[i].setArg(narg++, 0);
        err = all_kernels[i].setArg(narg++, b_all_res[i]);
        err = all_kernels[i].setArg(narg++, b_div_indexes[i]);
        if(err) {
            cout << "Error setting kernel arguments" << endl;
            return -1;
        }
    }

    const int index_argument = 3; //we need to keep changing argument 3

    // Now let's look how many times we need to send computations of BUFFER_SIZE
    const int n_times = max(n_events / BUFFER_SIZE / n_concurrent , 1);
    for(int k = 0; k < n_iter; k++) {
        cout << "Starting iteration " << k << endl;
        // Each of the concurrent one is sent in a differnt CPU thread
        // in parallel 
        double res = 0.0;
        double res2 = 0.0;
        vector<vector<double>> arr_res2(n_dim, vector<double>(BINS_MAX, 0.0));
        for (int i = 0; i < n_concurrent; i++) {
            for (int t = 0; t < n_times; t++) {
                cl::Event kevent;
                cl::Event wevent;

                // Send concurrent kernels
                const int rand_seed = k*n_concurrent*n_times + i*n_times + t;
                all_kernels[i].setArg(index_argument, rand_seed);
                if(err) {
                    cout << "Error setting index argument" << endl;
                    return -1;
                }

                // Enqueue writting the divisions 
                // err = q.enqueueMigrateMemObjects({b_divisions[i]}, 0);
                // for some unknown reason, the divisions need to be copied manualyl
                err = q.enqueueWriteBuffer(b_divisions[i], CL_TRUE, 0, sizeof(double)*div_size, divisions.data());
                if (err) { 
                    cout << "Error while copying divisions" << endl;
                    return -1;
                }

                // Now enqueue the kernels
                err = q.enqueueTask(all_kernels[i], NULL, &kevent);
                if (err) { 
                    cout << "Error while enqueuing kernel" << endl;
                    return -1;
                }

                // Now wait for this event to finish before asking for the data
                kevent.wait();
                err = q.enqueueMigrateMemObjects({b_all_res[i], b_div_indexes[i]}, CL_MIGRATE_MEM_OBJECT_HOST, NULL, &wevent);
                if (err) { 
                    cout << "Error while recovering data" << endl;
                    return -1;
                }
                wevent.wait();

                // Now we have the data, let us do something with it
                // in a critical way to first approximtion
                for (int b = 0; b < BUFFER_SIZE; b++) {
                    const double tmp = results[i][b];
                    const double tmp2 = tmp*tmp;
                    res += tmp;
                    res2 += tmp2;
                    const int bdx = b*n_dim;
                    for (int j = 0; j < n_dim; j++) {
                        const int idx = indexes[i][bdx+j];
                        arr_res2[j][idx] += tmp2;
                    }
                }
            }
        } 
        q.finish();
        // At this point we have collected all the results we need to recompute the grid so let's go
        const double err_tmp2 = max((n_events*res2 - res*res)/(n_events-1.0), 1e-30);
        const double sigma_tmp = sqrt(err_tmp2);
        printf("For iteration %d, result: %1.5f +- %1.5f\n", k+1, res, sigma_tmp);

        const double wgt_tmp = 1.0/pow(sigma_tmp, 2);
        total_res += res*wgt_tmp;
        total_weight += wgt_tmp;

        for (int j = 0; j < n_dim; j ++)
            refine_grid(arr_res2[j].data(), &divisions[j*BINS_MAX]);

        *vegas_result = total_res/total_weight;
        *sigma = sqrt(1.0/total_weight);
    }

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

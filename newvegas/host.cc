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

void random_bunch(aligned_vector<double> &randoms, const int n_dim) {
    // since it is just random we don't care about the order
    for (int i = 0; i < BUFFER_SIZE*n_dim; i++ ) {
        randoms[i] = (double)rand() / RAND_MAX;
    }
}

void rebin(const vector<double> &rw, const double rc, double *subdivisions) {
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

void refine_grid(const vector<double> &res_sq, double *subdivisions){
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

int vegas(std::string kernel_file, const int device_idx, const int warmup, const int n_dim, const int n_iter, const int n_events_raw, double *final_result, double *sigma) {

    const int n_threads = 1;
    const string kernel_name = "events_kernel";

    // 1. Needs to decide how many threads are going to be open in total
    // and how many events each thread is going to have
    // later each thread will run the number of events per kernel given by 
    // the BUFFER_SIZE variable the OCL kernel was compiled with
    // which in turn will define a number of kernels
    const int n_kernels = (int) n_events_raw/BUFFER_SIZE/n_threads;
    const int n_events = n_kernels*n_threads*BUFFER_SIZE; // make sure the number of events is a multiple of the number of threads and BUFFER_SIZE

    // 2. Initialize host-side device-usable variables 
    aligned_vector<double> divisions(n_dim*BINS_MAX);
    vector<vector<double>> arr_res2(n_dim, vector<double>(BINS_MAX));

    // 3. Initialize integration (fake init)
    srand(0);
    const double xjac = 1.0/n_events;
    vector<double> rw_tmp(BINS_MAX, 1.0); // fake init
    for (int j = 0; j < n_dim; j++) {
        divisions[j*BINS_MAX] = 1.0;
        rebin(rw_tmp, 1.0/BINS_MAX, &divisions[j*BINS_MAX]);
    }

    // 3. Initialize OCL
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

    // 4. Initialize host-side OCL usable variables
    // and device side buffers
    vector<aligned_vector<double>> randoms(n_threads);
    vector<aligned_vector<double>> results(n_threads);
    vector<aligned_vector<short>> indexes(n_threads);

    cl::Buffer b_divisions(context, CL_MEM_READ_ONLY, sizeof(double)*n_dim*BINS_MAX); // in
    vector<cl::Buffer> b_randoms(n_threads),
        b_results(n_threads), b_indexes(n_threads);

    vector<cl::Kernel> kernel(n_threads);

    #pragma omp parallel for
    for (int t = 0; t < n_threads; t++) {
        randoms[t] = aligned_vector<double>(BUFFER_SIZE*n_dim); // in
        results[t] = aligned_vector<double>(BUFFER_SIZE); // out
        indexes[t] = aligned_vector<short>(BUFFER_SIZE*n_dim); // out

        // generate the initial bunch of random numbers and start by copying them to the device
        random_bunch(randoms[t], n_dim);

        b_randoms[t] = cl::Buffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double)*BUFFER_SIZE*n_dim, randoms[t].data(), &err);
        b_results[t] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(double)*BUFFER_SIZE, results[t].data(), &err);
        b_indexes[t] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short)*BUFFER_SIZE*n_dim, indexes[t].data(), &err);

        if (err) {
            cout << "ERROR creating buffers for thread " << t << endl;
        }

        // Read the kernel out of the program
#ifdef FPGABUILD
        const string cuname = kernel_name + ":{" + kernel_name + "_" + to_string(t+1) + "}";
#else
        const string cuname = kernel_name;
#endif
        kernel[t] = cl::Kernel(program, cuname.c_str(), &err);
        if (err) {
            cout << "ERROR reading kernel: " << cuname << endl;
        }


        // 5. Now prepare the event kernel with the appropiate variables
        cl_uint narg = 0;
        err = kernel[t].setArg(narg++, b_divisions);
        err = kernel[t].setArg(narg++, b_randoms[t]);
        err = kernel[t].setArg(narg++, n_dim);
        err = kernel[t].setArg(narg++, b_results[t]);
        err = kernel[t].setArg(narg++, b_indexes[t]);
        if (err) {
            cout << "ERROR setting kernel arguments" << endl;
        }

    }
    if(err) return -1;


    double total_res = 0.0;
    double total_weight = 0.0;
    // 6. Now we can dive in to the iteration loop
    for (int k = 0; k < n_iter; k++){
        double res = 0.0;
        double res2 = 0.0;
        // Copy the divisions to the buffer, this is constant for all kernels so only need to be done once
        err = q.enqueueWriteBuffer(b_divisions, CL_TRUE, 0, sizeof(double)*BINS_MAX*n_dim, divisions.data());
        #pragma omp parallel for
        for (int t = 0; t < n_threads; t++) {
            if (err) continue;
            for (int i = 0; i < n_kernels; i++) {
                cl::Event revent, webent;
                vector<cl::Event> kevent(1);

                // a. launch the kernel
                err = q.enqueueTask(kernel[t], NULL, &kevent[0]);
                if (err) cout << "ERROR launching the kernel" << endl;

                // b. launch the retrieval of data 
                err = q.enqueueMigrateMemObjects({b_results[t], b_indexes[t]}, CL_MIGRATE_MEM_OBJECT_HOST, &kevent, &revent);
                if (err) cout << "ERROR reading data from device " << endl;

                // c // {a, c}: generate the new bunch of random data 
                if (k != n_iter && i != n_kernels) { // no more randoms are needed
                    random_bunch(randoms[t], n_dim);

                    // d // {b, future}: enqueue the copy of data to the device for when the kernel has finished running
                    err = q.enqueueWriteBuffer(b_randoms[t], CL_FALSE, 0, sizeof(double)*BUFFER_SIZE*n_dim, randoms[t].data(), &kevent, &webent);
                    if (err) cout << "ERROR writing randoms to device " << endl;
                }

                // e. wait until the copy of data has finished to beging accumulation of results
                revent.wait();
#ifndef FPGABUILD
                // TODO: this is not necessary in FPGA but it is in CPU??
                err = q.enqueueReadBuffer(b_results[t], CL_TRUE, 0, sizeof(double)*BUFFER_SIZE, results[t].data());
                if (err) cout << "Error reading resuts " << endl;
                err = q.enqueueReadBuffer(b_indexes[t], CL_TRUE, 0, sizeof(short)*BUFFER_SIZE*n_dim, indexes[t].data());
                if (err) cout << "Error reading indexes " << endl;
#endif

                if (err) break;

                // f. accumulate results (thread private or separate by thread)
                #pragma omp critical
                {
                    for (int b = 0; b < BUFFER_SIZE; b++) {
                        const double tmp = results[t][b];
                        const double tmp2 = pow(tmp, 2);
                        res += tmp;
                        res2 += tmp2;
                        for (int j = 0; j < n_dim; j++) {
                            const int jdx = j*BUFFER_SIZE + b;
                            const short idx = indexes[t][jdx];
                            arr_res2[j][idx] += tmp2;
                        }
                    }
                } // end critical

                if (k != n_iter && i != n_kernels) webent.wait(); 
            }
        }
        q.finish();
        if (err) return -1;

        // 7. Time to recompute the grid dim by dim
        for (int j = 0; j < n_dim; j++) {
            refine_grid(arr_res2[j], &divisions[j*BINS_MAX]);
        }

        // 8. And compute the cross section
        res *= xjac;
        res2 *= pow(xjac, 2);
        const double err_tmp2 = max((n_events*res2 - res*res)/(n_events-1.0), 1e-30);
        const double sigma_tmp = sqrt(err_tmp2);
        printf("For iteration %d, result: %1.5f +- %1.5f\n", k+1, res, sigma_tmp);

        const double wgt_tmp = 1.0/pow(sigma_tmp, 2);
        total_res += res*wgt_tmp;
        total_weight += wgt_tmp;

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

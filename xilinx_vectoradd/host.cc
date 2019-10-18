#include <vector>
#include <string>
#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <fstream>

#include "mccl.h"
#include "definitions.h"
#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h> // Necessary to use the pointers to HBM memory
#ifdef CUNITS
const int threads = CUNITS;
#else
const int threads = 1;
#endif

using namespace std;
template <typename T>
using aligned_vector = vector<T, aligned_allocator<T>>;

DOUBLE drand() {
    const int N = 1000; // random enough
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    unsigned long int rdtsc = ((unsigned long int)hi << 32) | lo;
    return (DOUBLE) (rdtsc % N)/N;
}

int run_kernel(const string kernel_name, const string kernel_file, const int n_events, const int device_idx) {
    // Declare input HOST arrays
    const int chunk_size = (int) n_events / threads;
    vector<aligned_vector<DOUBLE>> in_A(threads, aligned_vector<DOUBLE>(chunk_size));
    vector<aligned_vector<DOUBLE>> in_B(threads, aligned_vector<DOUBLE>(chunk_size));
    // Use random numbers for sums
    #pragma omp parallel for
    for (int i = 0; i < threads; i++) {
        for (int j = 0; j < chunk_size; j++) {
            in_A[i][j] = drand();
            in_B[i][j] = drand();
        }
    }

    // Declare many output HOST arrays
    vector<aligned_vector<DOUBLE>> out_C(threads, aligned_vector<DOUBLE>(chunk_size));

    // OpenCL initialization
    struct timeval start, stop;
    gettimeofday(&start, 0);
    int err;
    // Read the device 
    const auto device = get_default_device(device_idx); // platform 0 == GPU
    // Create context, queue and program
    cl::Context context( {device} );
    cl::CommandQueue q(context, device,  CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::Program program;
    if (device_idx == 1) {
        program = read_program_from_bin(kernel_file, context, device);
    } else {
        program = read_program_from_file(kernel_file, context, device);
    }

    // Read the kernels out of the program
    vector<cl::Kernel> all_kernels;

    // Buffer 
    vector<cl::Buffer> bin_A, bin_B, bout_C;

    // Kernel events
    vector<vector<cl::Event>> kevents(threads, vector<cl::Event>(1));

    for (int i = 0; i < threads; i++) {
        cout << endl;
        cout << " # Starting preparation for thread: " << i + 1 << endl;

        string cuname = kernel_name + ":{" + kernel_name + "_" + to_string(i+1) + "}";

        cl::Kernel kernel(program, cuname.c_str(), &err);
        if (err) {
            cout << "Error reading kernel" << endl;
            cout << " > Kernel name: " << kernel_name << endl;
            cout << " > Kernel file: " << kernel_file << endl;
            return -1;
        }
        all_kernels.push_back(kernel);

        // Copy-IN buffers
        cl::Buffer buffer_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(DOUBLE) * chunk_size, in_A[i].data(), &err);
        cl::Buffer buffer_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(DOUBLE) * chunk_size, in_B[i].data(), &err);
        // Copy-OUT buffers
        cl::Buffer buffer_C(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(DOUBLE) * chunk_size, out_C[i].data(), &err);
        // end OCL initialization
        if (err) {
            cout << "Some error while allocating buffers in the device" << endl;
            return -1;
        }
        bin_A.push_back(buffer_A);
        bin_B.push_back(buffer_B);
        bout_C.push_back(buffer_C);

        // Prepare the event kernel
        cl_uint narg = 0;
        err = all_kernels[i].setArg(narg++, bin_A[i]);
        err = all_kernels[i].setArg(narg++, bin_B[i]);
        err = all_kernels[i].setArg(narg++, bout_C[i]);
        if (err) {
            cout << "Some error while setting the kernel arguments, error code: " << err << endl;
            return -1;
        } else {
            cout << " > > Arguments created for thread: " << i+1 << endl;
        }
        // Enqueue the copy of the input vectors
        err = q.enqueueMigrateMemObjects({bin_A[i], bin_B[i]}, 0);
        if (err) { 
            cout << "Some error while copying memory to device" << endl;
            return -1;
        } else {
            cout << "Memory migrated" << endl;
        }
        if (kernel_name == "vector_add_simple") {
            err = q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(chunk_size), cl::NullRange, NULL, NULL);
        } else if (kernel_name == "vector_add_xilinx_repo") {
            err = all_kernels[i].setArg(narg++, chunk_size);
            if (err) {
                cout << "Some error while setting the kernel arguments, error code: " << err << endl;
                return -1;
            }
            err = q.enqueueTask(all_kernels[i], NULL, &kevents[i][0]);
        }
        if (err) { 
            cout << "Some error while running the kernel" << endl;
            return -1;
        }
    }

    // Now to first approximation simply wait until all the information has been retrieved
    #pragma omp parallel for
    for (int i = 0; i < threads; i++) {
        // note that this is the wrong way of doing this but it illustrates
        // how can we have a stream-like thing doing the accumulation in CPU while the
        // FPGA computes
        kevents[i][0].wait();
        err = q.enqueueMigrateMemObjects({bout_C[i]}, CL_MIGRATE_MEM_OBJECT_HOST);
        if (err) { 
            cout << "Some error while copying memory from device" << endl;
        }
    }

    q.finish();
    gettimeofday(&stop, 0);
    const double result = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*1e-6;
    printf("Finished running OCL kernel %s, took: %1.7f seconds\n",kernel_name.c_str(),  result);
    ofstream f;
    if (device_idx == 0) {
        f.open("gpu.time", ios_base::app);
    } else if (device_idx == 1) {
        f.open("fpga.time", ios_base::app);
    }
    f << n_events << " " << result << endl;
    f.close();

    cout << "Result checker: ";
    #pragma omp parallel for
    for(int j = 0; j < threads; j++) {
        for(int i = 0; i < chunk_size; i++) {
            DOUBLE host_result = in_A[j][i] + in_B[j][i];
            if (host_result != out_C[j][i]) {
                cout << "Wrong result for member " << j << " Host: " << host_result << " Device: " << out_C[j][i] << endl;
                err = 1;
            }
        }
    }
    if (!err) {
        cout << "passed!" << endl;
    }
    return err;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage %s kernel_name kernel_file events device\n", argv[0]);
        exit(0);
    }
    const string kernel_name = argv[1];
    const string kernel_file = argv[2];
    const int n_events = atoi(argv[3]);
    const int device_idx = atoi(argv[4]);

    int state = run_kernel(kernel_name, kernel_file, n_events, device_idx);

    if (state != 0) {
       printf("Something went wrong\n");
    }
    return 0;
}

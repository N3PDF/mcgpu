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

DOUBLE drand() {
    const int N = 1000; // random enough
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    unsigned long int rdtsc = ((unsigned long int)hi << 32) | lo;
    return (DOUBLE) (rdtsc % N)/N;
}

int run_kernel(const string kernel_name, const string kernel_file, const int n_events, const int device_idx) {
    // Declare HOST arrays
    aligned_vector<DOUBLE> A(n_events);
    aligned_vector<DOUBLE> B(n_events);
    aligned_vector<DOUBLE> C(n_events);

    // Use random numbers for sums
    #pragma omp parallel for
    for (int i = 0; i < n_events; i ++) {
        A[i] = drand();
        B[i] = drand();
    }

    // OpenCL initialization
    struct timeval start, stop;
    gettimeofday(&start, 0);
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
    cl::Kernel kernel(program, kernel_name.c_str(), &err);
    if (err) {
        cout << "Error reading kernel" << endl;
        cout << " > Kernel name: " << kernel_name << endl;
        cout << " > Kernel file: " << kernel_file << endl;
        return -1;
    }


    // Now allocate the memory buffers on the device (not using CL_MEM_COPY_HOST_PTR in order to follow xilinx stupid implementation)
    // Copy-IN buffers
    cl::Buffer buffer_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(DOUBLE) * n_events, A.data(), &err);
    cl::Buffer buffer_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(DOUBLE) * n_events, B.data(), &err);
    // Copy-OUT buffers
    cl::Buffer buffer_C(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(DOUBLE) * n_events, C.data(), &err);
    // end OCL initialization
    if (err) {
        cout << "Some error while allocating buffers in the device" << endl;
        return -1;
    }

    // Prepare the event kernel
    cl_uint narg = 0;
    err = kernel.setArg(narg++, buffer_A);
    err = kernel.setArg(narg++, buffer_B);
    err = kernel.setArg(narg++, buffer_C);
    if (err) {
        cout << "Some error while setting the kernel arguments" << err << endl;
        return -1;
    }

    // Enqueue the copy of the input vectors
    err = q.enqueueMigrateMemObjects({buffer_A, buffer_B}, 0);
    if (err) { 
        cout << "Some error while copying memory to device" << endl;
        return -1;
    }

    /* Here start the kernel dependent lines
     * up to here everything should be shared by all kernels since 
     * this is just a copy of two arrays (A and B) into a vector C
     */

    if (kernel_name == "vector_add_simple") {
        err = q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n_events), cl::NullRange, NULL, NULL);
    } else if (kernel_name == "vector_add_xilinx_repo") {
        err = kernel.setArg(narg++, n_events);
        err = q.enqueueTask(kernel);
    }
    if (err) { 
        cout << "Some error while running the kernel" << endl;
        return -1;
    }

    /////// END
    err = q.enqueueMigrateMemObjects({buffer_C}, CL_MIGRATE_MEM_OBJECT_HOST);
    if (err) { 
        cout << "Some error while copying memory from device" << endl;
        return -1;
    }
    q.finish();
    gettimeofday(&stop, 0);
    const double result = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*1e-6;
    printf("Finished running OCL kernel, took: %1.7f seconds\n", result);

    cout << "Result checker: ";
    for(int i = 0; i < n_events; i++) {
        DOUBLE host_result = A[i] + B[i];
        if (host_result != C[i]) {
            cout << "Wrong result for member " << i << " Host: " << host_result << " Device: " << C[i] << endl;
            err = 1;
            return - 1;
        }
    }
    cout << "passed!" << endl;
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

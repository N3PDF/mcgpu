#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include <sys/param.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// use boost.compute 
int read_file(unsigned char **output, size_t *size, const char *name) {
    FILE* fp = fopen(name, "rb");
    if (!fp) {
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *output = (unsigned char*) malloc(*size);
    if (!*output) {
        fclose(fp);
        return -1;
    }

    fread(*output, *size, 1, fp);
    fclose(fp);
    return 0;
}

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
            int int_xn = MAX(0, MIN( (int) xn, BINS_MAX));
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

int vegas(std::string binaryFile, const int warmup, const int n_dim, const int n_iter, const int n_events, double *final_result, double *sigma) {

    // Do OCL initialization
    cl_platform_id platform;
    cl_int err = clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    // create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueueWithProperties(context, device, 0, NULL);

//    //Creating Context and Command Queue for selected Device
//    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
//    OCL_CHECK(
//        err,
//        cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
//    OCL_CHECK(err,
//              std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
//    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    // read_binary() command will find the OpenCL binary file created using the
    // xocc compiler load into OpenCL Binary and return a pointer to file buffer
    // and it can contain many functions which can be executed on the
    // device.
//    auto fileBuf = xcl::read_binary_file(binaryFile);
//    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
//    devices.resize(1);
//    cl_program program = clCreateProgramWithBinary(context, 1, 
//    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // This call will extract a kernel out of the program we loaded
//    OCL_CHECK(err, cl::Kernel events_kernel(program, "events_kernel", &err));

    // Now create the program by reading it up to a string and then creating it with opencl
    unsigned char* program_file = NULL;
    size_t program_size = 0;
    read_file(&program_file, &program_size, "kernel.cl");

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&program_file, &program_size, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    free(program_file);

    // Red the kernel out of the program
    cl_kernel kernel = clCreateKernel(program, "events_kernel", &err);


    // Now allocate the memory buffers on the device
    cl_mem buffer_all_randoms = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*(n_dim*n_events), NULL, &err);
    cl_mem buffer_all_xwgts = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*(n_dim*n_events), NULL, &err);
    cl_mem buffer_all_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*n_events, NULL, &err);
    cl_mem buffer_all_res2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*n_events, NULL, &err);

//    OCL_CHECK(err,
//            cl::Buffer buffer_all_randoms(context,
//                                CL_MEM_READ_ONLY,
//                                sizeof(double) * (n_dim*n_events),
//                                NULL,
//                                &err));
//    OCL_CHECK(err,
//            cl::Buffer buffer_all_xwgts(context,
//                                CL_MEM_READ_ONLY,
//                                sizeof(int) * (n_dim*n_events),
//                                NULL,
//                                &err));
//    OCL_CHECK(err,
//            cl::Buffer buffer_all_res(context,
//                                    CL_MEM_WRITE_ONLY,
//                                    sizeof(double) * n_events,
//                                    NULL,
//                                    &err));
//
//    OCL_CHECK(err,
//            cl::Buffer buffer_all_res2(context,
//                                    CL_MEM_WRITE_ONLY,
//                                    sizeof(double) * n_events,
//                                    NULL,
//                                    &err));
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
        double res = 0.;
        double res2 = 0.;
        double sigma = 0.;

        // input arrays
        int NN = n_dim*n_events;
        //double *all_randoms = new double[NN];
        double *all_randoms = (double*) aligned_alloc(NN, NN * sizeof(double));
        int *all_div_indexes = (int*) aligned_alloc(NN, NN * sizeof(int));
        for (int i = 0; i < NN; i++) all_div_indexes[i] = 0;

        // output arrays
        double *all_res = (double*) aligned_alloc(n_events, n_events * sizeof(double));
        double *all_res2 = (double*) aligned_alloc(n_events, n_events * sizeof(double));

        double *all_xwgts = generate_random_array(n_events, n_dim, divisions, all_randoms, all_div_indexes);

        cl_event wb_events[2];
        err = clEnqueueWriteBuffer(q, buffer_all_randoms, CL_FALSE, 0, NN*sizeof(double), all_randoms, 0, NULL, &wb_events[0]);
        err = clEnqueueWriteBuffer(q, buffer_all_xwgts, CL_FALSE, 0, n_events*sizeof(double), all_xwgts, 0, NULL, &wb_events[1]);


//        OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_all_randoms, CL_TRUE, 0, NN*sizeof(double), all_randoms));
//        OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_all_xwgts, CL_TRUE, 0, n_events*sizeof(double), all_xwgts));

         //set the kernel Arguments
        int narg = 0;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_all_randoms);
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_all_xwgts);
        err = clSetKernelArg(kernel, 2, sizeof(int), &n_dim);
        err = clSetKernelArg(kernel, 3, sizeof(int), &n_events);
        err = clSetKernelArg(kernel, 4, sizeof(double), &xjac);
        err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &buffer_all_res);
        err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &buffer_all_res2);
//        OCL_CHECK(err, err = events_kernel.setArg(narg++, buffer_all_randoms));
//        OCL_CHECK(err, err = events_kernel.setArg(narg++, buffer_all_xwgts));
//        OCL_CHECK(err, err = events_kernel.setArg(narg++, n_dim));
//        OCL_CHECK(err, err = events_kernel.setArg(narg++, n_events));
//        OCL_CHECK(err, err = events_kernel.setArg(narg++, xjac));
//        OCL_CHECK(err, err = events_kernel.setArg(narg++, buffer_all_res));
//        OCL_CHECK(err, err = events_kernel.setArg(narg++, buffer_all_res2));

        // These commands will load the all_randoms and all_xwgts from the host
        // application and into the buffer_all_randoms and buffer_all_xwgts cl::Buffer objects. The data
        // will be be transferred from system memory over PCIe to the FPGA on-board
        // DDR memory.
//        OCL_CHECK(err,
//                err = q.enqueueMigrateMemObjects({buffer_all_randoms, buffer_all_xwgts},
//                                                0));

        // Launch the Kernel
        const size_t global_offset = 0;
        //const size_t snevents = (size_t) n_events;
        const size_t snevents = 100000;
        const size_t local_work = 1000;
        cl_event kernel_event;
        err = clEnqueueNDRangeKernel(q, kernel, 1, &global_offset, &snevents, &local_work, 2, wb_events, &kernel_event);
//        OCL_CHECK(err, err = q.enqueueNDRangeKernel(events_kernel, cl::NullRange, cl::NDRange(n_events)));

        // copy result from device to program
        err = clEnqueueReadBuffer(q, buffer_all_res, CL_TRUE, 0, n_events*sizeof(double), all_res, 1, &kernel_event, NULL);
        err = clEnqueueReadBuffer(q, buffer_all_res2, CL_TRUE, 0, n_events*sizeof(double), all_res2, 1, &kernel_event, NULL);
//        OCL_CHECK(err,
//                err = q.enqueueMigrateMemObjects({buffer_all_res, buffer_all_res2},
//                                                CL_MIGRATE_MEM_OBJECT_HOST));

        //q.finish();
        err = clFinish(q);
//        OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_all_res, CL_TRUE, 0, n_events*sizeof(double), all_res));
//        OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_all_res2, CL_TRUE, 0, n_events*sizeof(double), all_res2));

        for (int i = 0; i < n_events; i++)
        {
            res += all_res[i];
            res2 += all_res2[i];
        }

        double *arr_res2 = new double[n_dim*BINS_MAX]();
        for( int i = 0; i < n_events; i++ )
            for (int j = 0; j < n_dim; j++)
                arr_res2[j*BINS_MAX + all_div_indexes[i*n_dim + j]] += all_res2[i];

        double err_tmp2 = MAX((n_events*res2 - res*res)/(n_events-1.0), 1e-30);
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

        clReleaseMemObject(buffer_all_res);
        clReleaseMemObject(buffer_all_res2);
        clReleaseMemObject(buffer_all_xwgts);
        clReleaseMemObject(buffer_all_randoms);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(q);
        clReleaseContext(context);
        clReleaseDevice(device);


    delete[] divisions;

    return 0;
}

int main(int argc, char **argv) {

    if (argc < 2) {
        fprintf(stderr, "usage %s number_of_events number_of_dimensions\n", argv[0]);
        exit(0);
    }
    int n_events = atoi(argv[1]);
    int n_dim = atoi(argv[2]);
    int n_iter = 5;

    std::string binaryFile = "host.cc";
    // compute the size of array in bytes

//    int n_dim = 4;
//    int n_events = (int) 1e6;
//    int n_iter = 5;

    double res, sigma;
    struct timeval start, stop;

    gettimeofday(&start, 0);
    int state = vegas(binaryFile, 1, n_dim, n_iter, n_events, &res, &sigma);
    gettimeofday(&stop, 0);

    if (state != 0) {
       printf("Something went wrong\n");
    }

    printf("Final result: %1.5f +- %1.5f\n", res, sigma);
    double result = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*1e-6;
    printf("It took: %1.7f seconds\n", result);

    return 0;
}

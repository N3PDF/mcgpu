#include "xcl2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/param.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BINS_MAX 50
#define ALPHA 1.5

double internal_rand(){
    double x = (double) rand()/RAND_MAX;
    return x;
}

double* generate_random_array(const int n_events, const int n_dim, const double *divisions, double *x, int *div_index) {

    double *all_wgts = new double[n_events]();
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

    // init OCL device
    cl_int err;

    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];

    //Creating Context and Command Queue for selected Device
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(
        err,
        cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err,
              std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    // read_binary() command will find the OpenCL binary file created using the
    // xocc compiler load into OpenCL Binary and return a pointer to file buffer
    // and it can contain many functions which can be executed on the
    // device.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    /*
    // These commands will allocate memory on the FPGA. The cl::Buffer objects can
    // be used to reference the memory locations on the device. The cl::Buffer
    // object cannot be referenced directly and must be passed to other OpenCL
    // functions.
    OCL_CHECK(err,
              cl::Buffer buffer_a(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  size_in_bytes,
                                  source_a.data(),
                                  &err));
    OCL_CHECK(err,
              cl::Buffer buffer_b(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  size_in_bytes,
                                  source_b.data(),
                                  &err));
    OCL_CHECK(err,
              cl::Buffer buffer_result(context,
                                       CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                       size_in_bytes,
                                       source_results.data(),
                                       &err));

    // This call will extract a kernel out of the program we loaded in the
    // previous line. A kernel is an OpenCL function that is executed on the
    // FPGA. This function is defined in the src/vetor_addition.cl file.
    OCL_CHECK(err, cl::Kernel krnl_vector_add(program, "vector_add", &err));

    //set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, buffer_result));
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, buffer_a));
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, buffer_b));
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, DATA_SIZE));

    // These commands will load the source_a and source_b vectors from the host
    // application and into the buffer_a and buffer_b cl::Buffer objects. The data
    // will be be transferred from system memory over PCIe to the FPGA on-board
    // DDR memory.
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_a, buffer_b},
                                               0));

    //Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will write the data from the
    // buffer_result cl_mem object to the source_results vector
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_result},
                                               CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    */

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
        double *all_randoms = new double[NN];
        int *all_div_indexes = new int[NN]();

        // output arrays
        double *all_res = new double[n_events];
        double *all_res2 = new double[n_events];

        double *all_xwgts = generate_random_array(n_events, n_dim, divisions, all_randoms, all_div_indexes);

        for ( int i = 0;  i < n_events; i++ ) {
            double wgt = all_xwgts[i]*xjac;
            double a = 0.1;
            double pref = pow(1.0/a/sqrt(M_PI), n_dim);
            double coef = 0.0;
            for (int j = 0; j < n_dim; j++) {
                coef += pow( (all_randoms[i*n_dim + j] - 1.0/2.0)/a, 2 );
            }
            double lepage = pref*exp(-coef);

            double tmp = wgt*lepage;
            all_res[i] = tmp;
            all_res2[i] = pow(tmp,2);
        }

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

        delete[] all_randoms;
        delete[] all_div_indexes;
        delete[] all_xwgts;
        delete[] all_res;
        delete[] all_res2;
        delete[] arr_res2;
    }

    *final_result = total_res/total_weight;
    *sigma = sqrt(1.0/total_weight);

    delete[] divisions;

    return 0;
}



// This example illustrates the very simple OpenCL example that performs
// an addition on two vectors
int main(int argc, char **argv) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    // compute the size of array in bytes

    int n_dim = 4;
    int n_events = (int) 1e6;
    int n_iter = 5;

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
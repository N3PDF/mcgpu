#include "definitions.h"


// Copied form https://www.xilinx.com/html_docs/xilinx2017_4/sdaccel_doc/izc1504034357785.html
__kernel
void vector_add_simple(__global const DOUBLE *a, __global const DOUBLE *b, __global DOUBLE *c) {
    int idx = get_global_id(0);
    c[idx] = a[idx] + b[idx];
}


// Copied from Xilinx repo: getting_started/hello_world/helloworld_ocl/src/vector_addition.clm  
// modified just to take the arguments in the correct order and to take DOUBLE data


#define BUFFER_SIZE 1024
#define DATA_SIZE 4096
//TRIPCOUNT indentifier
__constant uint c_len = DATA_SIZE/BUFFER_SIZE;
__constant uint c_size = BUFFER_SIZE;

void copy_data(DOUBLE* arr_to, __global const DOUBLE* arr_from, const int size) {
#ifdef FPGABUILD
    __attribute__((xcl_loop_tripcount(c_size, c_size)))
    __attribute__((xcl_pipeline_loop(1)))
#endif
    for (int i = 0; i < size; i++){
        arr_to[i] = arr_from[i];
    }
}

void computation(__global DOUBLE* result, const DOUBLE* arr1, const DOUBLE* arr2, const int size) {
#ifdef FPGABUILD
    __attribute__((xcl_loop_tripcount(c_size, c_size)))
    __attribute__((xcl_pipeline_loop(1)))
#endif
    for (int i = 0; i < size; i++){
        result[i] = arr1[i] + arr2[i];
    }
}

__kernel
__attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__ ((xcl_dataflow)) 
void vector_add_xilinx_repo(global const DOUBLE* a,
                global const DOUBLE* b,
                global  DOUBLE* c,
                       const int n_elements) {
    DOUBLE arrayA[BUFFER_SIZE];
    DOUBLE arrayB[BUFFER_SIZE];

#ifdef FPGABUILD
    __attribute__((xcl_loop_tripcount(c_len, c_len)))
#endif
    for (int i = 0 ; i < n_elements ; i += BUFFER_SIZE) {
        int size = BUFFER_SIZE;
        
        if (i + size > n_elements) size = n_elements - i;

        copy_data(arrayA, &a[i], BUFFER_SIZE);
        copy_data(arrayB, &b[i], BUFFER_SIZE);

        computation(&c[i], arrayA, arrayB, BUFFER_SIZE);
    }
}

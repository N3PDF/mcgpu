#pragma once

// Xilinx needs the OpenCL to be forced at 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
// The following is necessasry for the Xilinx way of constructing the program from binary
// some testing is needed, but it might be avoided
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1 
#include <CL/cl2.hpp> 

// Generic utilities
std::string read_file_to_str(const std::string &file_name);

// Device and platform selection wrappers
cl::Platform platform_selection(const int sel);
cl::Device device_selection(const int platform_sel, const int device_sel);
cl::Device get_default_device(const int platform_sel);

// Kernel loading wrappers
cl::Program read_program_from_file(const std::string &file_name, cl::Context ctx, cl::Device);
cl::Program read_program_from_bin(const std::string &file_name, cl::Context ctx, cl::Device device);

// This allocator is copied verbatim from Xilinx's repository
//////// --- BEGIN COPY
// When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood                                                                                   
// User ptr is used if and only if it is properly aligned (page aligned). When not                                                                                  
// aligned, runtime has no choice but to create its own host side buffer that backs                                                                                 
// user ptr. This in turn implies that all operations that move data to and from                                                                                    
// device incur an extra memcpy to move data to/from runtime's own host buffer                                                                                      
// from/to user pointer. So it is recommended to use this allocator if user wish to                                                                                 
// Create Buffer/Memory Object with CL_MEM_USE_HOST_PTR to align user buffer to the                                                                                 
// page boundary. It will ensure that user buffer will be used when user create                                                                                     
// Buffer/Mem Object with CL_MEM_USE_HOST_PTR.                                                                                                                      
template <typename T>                                                                                                                                               
struct aligned_allocator                                                                                                                                            
{                                                                                                                                                                   
    using value_type = T;                                                                                                                                             
    T* allocate(std::size_t num)                                                                                                                                      
    {                                                                                                                                                                 
        void* ptr = nullptr;                                                                                                                                            
        if (posix_memalign(&ptr,4096,num*sizeof(T)))                                                                                                                    
            throw std::bad_alloc();                                                                                                                                       
        return reinterpret_cast<T*>(ptr);                                                                                                                               
    }                                                                                                                                                                 
    void deallocate(T* p, std::size_t num)                                                                                                                            
    {                                                                                                                                                                 
        free(p);                                                                                                                                                        
    }                                                                                                                                                                 
};    
//////// --- END COPY

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


// HBM Management
#define MAX_HBM_BANKCOUNT 32
#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
const int bank[MAX_HBM_BANKCOUNT] = {
    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
    BANK_NAME(30), BANK_NAME(31)};

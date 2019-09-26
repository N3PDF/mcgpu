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
cl::Platform platform_selection();
cl::Device device_selection(const int platform_sel, const int device_sel);
cl::Device get_default_device(const int platform_sel);

// Kernel loading wrappers
cl::Program read_program_from_file(const std::string &file_name, cl::Context ctx, cl::Device);
cl::Program read_program_from_bin(const std::string &file_name, cl::Context ctx, cl::Device device);

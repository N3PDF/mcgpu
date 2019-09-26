#pragma once

// Xilinx needs the OpenCL to be forced at 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#include <CL/cl2.hpp> 

std::string read_file_to_str(const std::string &file_name);
cl::Program read_program_from_file(const std::string &file_name, cl::Context ctx);

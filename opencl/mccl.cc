#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "mccl.h"

using namespace std;

string read_file_to_str(const string &file_name) {
    /* Reads a kernel file into a string */
    cout << "Reading " << file_name << endl;
    ifstream filein(file_name);
    stringstream buffer;
    buffer << filein.rdbuf();
    return buffer.str();
}


cl::Program read_program_from_file(const string &file_name, cl::Context ctx) {
    string full_kernel = read_file_to_str(file_name);
    cl::Program::Sources sources;
    sources.push_back({full_kernel.c_str(), full_kernel.length()});
    cl::Program program(ctx, sources);
    return program;
}

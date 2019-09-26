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

cl::Platform platform_selection(const int sel) {
    /* Finds all platforms available in the system
     * and asks the user which one wants to use
     *
     * If the argument 'sel' is set, returns that
     * platform back
     */
    vector<cl::Platform> platforms;
    int err = cl::Platform::get(&platforms);
    int total = (int) platforms.size();
    cout << "Found " << total << " platforms:" << endl;
    for (int i = 0; i < total; i++) {
        cout << "[" << i << "] " << platforms[i].getInfo<CL_PLATFORM_NAME>(&err) << endl;
    }
    if (sel < 0) {
        int mine = -1;
        while (mine >= total || mine < 0) { 
            cout << "Which platform do you want to use? ([" << 0 << "-" << total-1 <<"]): ";
            cin >> mine;
        }
        return platforms[mine];
    } else {
        return platforms[sel];
    }
}
cl::Platform platform_selection() { return platform_selection(-1); };

cl::Device device_selection(const int platform_sel, const int device_sel) {
    /* Wrapper around platform_selection to obtain directly a device */
    auto platform = platform_selection(platform_sel);
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    return devices[device_sel];
}
cl::Device get_default_device(const int platform_sel) {
    auto platform = platform_selection(platform_sel);
    return device_selection(platform_sel, 0);
}

cl::Program read_program_from_file(const string &file_name, cl::Context ctx, cl::Device device) {
    /* Wrapper around OpenCL program creation to read and build a kernel file
     * for a given device 
     */
    string full_kernel = read_file_to_str(file_name);
    cl::Program::Sources sources;
    sources.push_back({full_kernel.c_str(), full_kernel.length()});
    cl::Program program(ctx, sources);
    if(program.build({device}) != CL_SUCCESS) {
        cout << "There was an error when building the program" << endl;
        cout << "Log info: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        exit(1);
    }
    return program;
}
cl::Program read_program_from_bin(const string &file_name, cl::Context ctx, cl::Device device) {
    /* Wrapper around OpenCL program creation to read a binary kernel */
    // Copied from xilinx header because the compilation is not working for me????
    std::ifstream bin_file(file_name.c_str(), ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    auto nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    vector<unsigned char> buf;
    buf.resize(nb);
    bin_file.read(reinterpret_cast<char*>(buf.data()), nb);
    cl::Program::Binaries bins{{buf.data(), buf.size()}};
    int err;
    vector<cl::Device> devices = {device};
    cl::Program program(ctx, devices, bins, NULL, &err);
    return program;
}

.PHONY: help

# Running options
EVENTS=4000
DIMENSIONS=4
TARGET:=sw_emu
PROFILE:=no
DEBUG:=no
DEVICE:=xilinx_u280_xdma_201920_1

ECHO:= @echo
help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make xilinx TARGET=<sw_emu/hw_emu/hw>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) " 	   (default $(TARGET))"
	$(ECHO) ""
	$(ECHO) "  make driver"
	$(ECHO) "      Command to generate the driver host code."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) " Options: "
	$(ECHO) "    EVENTS how many events to run (default $(EVENTS))"
	$(ECHO) "    DIMENSIONS how many dimensions to run (default $(DIMENSIONS))"
	$(ECHO) "    PROFILE enable profiling (default $(PROFILE))"
	$(ECHO) "    DEBUG enable debug summary report (default $(DEBUG))"
	$(ECHO) ""
	$(ECHO) ""
	$(ECHO) ""

HOST_SRCS = host.cc mccl.cc
EXECUTABLE = driver
KERNEL_NAME = events_kernel
KERNEL_FILE = kernel.cpp

## Select the compiler
CXX := g++
VPP := v++

## Generate the building folders
device2dsa = $(strip $(patsubst %.xpfm, % , $(shell basename $(DEVICE))))
#   device2dsa - create a filesystem friendly name from device name
#   $(1) - full name of device
XSA := $(call device2dsa, $(DEVICE))
TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

## Set up the OpenCL libraries
# env: 
# 	XILINX_XRT: /opt/xilinx/xrt
# 	XILINX_VIVADO: /opt/Xilinx/Vivado/2019.2
OPENCL_INCLUDE := $(XILINX_XRT)/include
VIVADO_INCLUDE := $(XILINX_VIVADO)/include
OPENCL_LIB := $(XILINX_XRT)/lib
opencl_CXXFLAGS=-I$(OPENCL_INCLUDE) -I$(VIVADO_INCLUDE)
opencl_LDFLAGS=-L$(OPENCL_LIB) -lOpenCL -lpthread

## Set up openmp
# OPENMP = -fopenmp

## Set up the host compiler flags
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O4 -std=c++14 $(OPENMP) -fmessage-length=0
LDFLAGS += $(opencl_LDFLAGS) -lrt -lstdc++

## Set up the kernel compiler flags
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps

## Set up the name of the compiled kernel and output files
KERNEL_BIN = $(BUILD_DIR)/$(KERNEL_NAME).xclbin
KERNEL_LIB = $(TEMP_DIR)/$(KERNEL_NAME).xo
EMCONFIG_DIR = $(TEMP_DIR)

########## Rules
.PHONY: xilinx clean cleanall emconfig exe

exe: $(EXECUTABLE)

xilinx: $(exe) $(KERNEL_BIN) emconfig

build: $(KERNEL_BIN)

$(KERNEL_LIB): $(KERNEL_FILE)
	mkdir -p $(TEMP_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k $(KERNEL_NAME) -I'$(<D)' -o'$@' '$<'

$(KERNEL_BIN): $(KERNEL_LIB)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR) -l $(LDCLFLAGS) -o'$@' $(+)

$(EXECUTABLE): $(HOST_SRCS)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) -o '$@' $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

run: xilinx $(EXECUTABLE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	cp -rf $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(EVENTS) $(DIMENSIONS) $(KERNEL_BIN) 1
else
	./$(EXECUTABLE) $(EVENTS) $(DIMENSIONS) $(KERNEL_BIN) 1
endif


#### Cleans
RMDIR=rm -rf
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) _x.* *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin

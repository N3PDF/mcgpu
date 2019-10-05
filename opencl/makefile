.PHONY: help

# Running options
EVENTS=10000000
DIMENSIONS=7
TARGET:=sw_emu
PROFILE:=no
DEBUG:=no
DEVICE:=xilinx_u280_xdma_201910_1

ECHO:= @echo
help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make run-gpu"
	$(ECHO) "      compiles and run the CPP program in GPU"
	$(ECHO) ""
	$(ECHO) "  make run-cpu"
	$(ECHO) "      compiles and run the CPP program in CPU"
	$(ECHO) ""
	$(ECHO) "  make run-python"
	$(ECHO) "      compiles and run the pyopencl version of the program "
	$(ECHO) ""
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
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw>"
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

# Compiler selection
PYTHON = python3
CXX = g++
CXXFLAGS = -lOpenCL -Wall -std=c++14
ifeq ($(DEBUG), yes)
CXXFLAGS += -Og -g
else
CXXFLAGS += -O4
endif
XXX = $(XILINX_SDX)/bin/xcpp
XOCC = $(XILINX_SDX)/bin/xocc

# Some targets
TARGET_CPP = cpp-opencl
XCLBIN = ./xclbin
KERNEL_SRC = kernel.cl
KERNEL_NAMES = events_kernel # generate_random_array_kernel
KERNEL_BINS = $(addprefix $(XCLBIN)/$(TARGET)_, $(addsuffix .xo, $(KERNEL_NAMES)))

# C source files
HOST_SRCS = host.cc mccl.cc

##########################
### Xilinx specific options
device2dsa = $(strip $(patsubst %.xpfm, % , $(shell basename $(DEVICE))))
#   device2dsa - create a filesystem friendly name from device name
#   $(1) - full name of device
DSA := $(call device2dsa, $(DEVICE))
BUILD_DIR := ./_x.$(TARGET).$(DSA)

BUILD_DIR_bitstream = $(BUILD_DIR)/bitstream
# Xilinx specific libraries and includes (note that for xilinx opencl comes from a different place)
OPENCL_INCLUDE:= $(XILINX_XRT)/include/
VIVADO_INCLUDE:= $(XILINX_VIVADO)/include/

opencl_CXXFLAGS=-I$(OPENCL_INCLUDE) -I$(VIVADO_INCLUDE)

OPENCL_LIB:=$(XILINX_XRT)/lib/
opencl_LDFLAGS=-L$(OPENCL_LIB) -lOpenCL -lpthread

CXXFLAGS += $(opencl_CXXFLAGS)
LDFLAGS += $(opencl_LDFLAGS)

# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++

# Kernel compiler global settings
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps

EXECUTABLE = driver
CMD_ARGS = $(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin

EMCONFIG_DIR = $(XCLBIN)/$(DSA)

BINARY_CONTAINERS += $(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin
BINARY_CONTAINER_bitstream_OBJS += $(XCLBIN)/bitstream.$(TARGET).$(DSA).xo

##########################

#Generates profile summary report
ifeq ($(PROFILE), yes)
LDCLFLAGS += --profile_kernel data:all:all:all
endif

#Generates debug summary report
ifeq ($(DEBUG), yes)
CLFLAGS += --dk protocol:all:all:all
endif

##### if making driver or xilinx, check the correct enviromental variables are set
#Checks for XILINX_SDX
# ifndef XILINX_SDX
# $(error XILINX_SDX variable is not set, please set correctly and rerun)
# endif
# #Checks for XILINX_XRT
# check-xrt:
# ifndef XILINX_XRT
# 	$(error XILINX_XRT variable is not set, please set correctly and rerun)
# endif
# check-devices:
# ifndef DEVICE
# 	$(error DEVICE not set. Please set the DEVICE properly and rerun. Run "make help" for more details.)
# endif
# check-aws_repo:
# ifndef SDACCEL_DIR
# 	$(error SDACCEL_DIR not set. Please set it properly and rerun. Run "make help" for more details.)
# endif
########


.PHONY: xilinx clean cleanall emconfig exe
exe: $(EXECUTABLE)
xilinx: $(exe) $(BINARY_CONTAINERS) emconfig


build: $(BINARY_CONTAINERS)

########################################################## MORE XILINX SPECIFIC STUFF
# Building kernel
$(XCLBIN)/$(TARGET)_%.xo: $(KERNEL_SRC)
	@echo > Compiling opencl source
	@mkdir -p $(XCLBIN)
	@$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_bitstream) -c -k $* -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin: $(KERNEL_BINS)
	@echo > Building binary kernel for $(TARGET)
	@mkdir -p $(XCLBIN)
	@$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_bitstream) -l $(LDCLFLAGS) -o'$@' $(+)

# $(XCLBIN)/bitstream.$(TARGET).$(DSA).xo: kernel.cl
# 	mkdir -p $(XCLBIN)
# 	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_bitstream) -c -k events_kernel -I'$(<D)' -o'$@' '$<'
# $(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin: $(BINARY_CONTAINER_bitstream_OBJS)
# 	mkdir -p $(XCLBIN)
# 	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_bitstream) -l $(LDCLFLAGS) --nk events_kernel:1 -o'$@' $(+)

# Building Host
$(EXECUTABLE): $(HOST_SRCS) $(HOST_HDRS)
	$(XXX) $(CXXFLAGS) $(HOST_SRCS) $(HOST_HDRS) -o '$@' $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

check: xilinx driver
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	cp -rf $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) 10000 4 $(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin 1
else
	 ./$(EXECUTABLE) $(EVENTS) $(DIMENSIONS) $(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin 1
endif
# 	sdx_analyze profile -i profile_summary.csv -f html
#################################################################################

$(TARGET_CPP): $(HOST_SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: run-gpu run-python run-cpu

run-cpu: $(TARGET_CPP)
	./$(TARGET_CPP) $(EVENTS) $(DIMENSIONS) $(KERNEL_SRC) 2

run-gpu: $(TARGET_CPP)
	./$(TARGET_CPP) $(EVENTS) $(DIMENSIONS) $(KERNEL_SRC) 0

run-python:
	$(PYTHON) vegas_mc_pyopencl.py -n $(EVENTS) -d $(DIMENSIONS)

# Cleaning stuff
RMDIR=rm -rf
clean:
	-$(RMDIR) $(TARGET_CPP)
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*}
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv
	-$(RMDIR) src/*.ll _xocc_* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) _x.*

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) _x.*

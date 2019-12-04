.PHONY: help

# Running options
EVENTS=100000
TARGET:=sw_emu
PROFILE:=no
DEBUG:=no
DEVICE:=xilinx_u280_xdma_201910_1
KERNEL:=vector_add_xilinx_repo
#vector_add_simple
USERAM:=yes
CUNITS:=2

ECHO:= @echo
help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make run-gpu"
	$(ECHO) "      compiles and run the CPP program in GPU"
	$(ECHO) ""
	$(ECHO) "  make run-cpu"
	$(ECHO) "      compiles and run the CPP program in CPU"
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
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) " Options: "
	$(ECHO) "    KERNEL kernel to run (default $(KERNEL))"
	$(ECHO) "    EVENTS how many events to run (default $(EVENTS))"
	$(ECHO) "    PROFILE enable profiling (default $(PROFILE))"
	$(ECHO) "    DEBUG enable debug summary report (default $(DEBUG))"
	$(ECHO) "    USERAM enable using RAM instead of HBM $(USERAM)"
	$(ECHO) "    CUNITS number of computings units, default: $(CUNITS)"
	$(ECHO) ""
	$(ECHO) ""
	$(ECHO) ""

# Compiler selection
PYTHON = python3
CXX = g++
CXXFLAGS = -lOpenCL -Wall -std=c++14 -fopenmp -DCUNITS=$(CUNITS)
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
KERNEL_BINS = $(addprefix $(XCLBIN)/$(TARGET)_, $(addsuffix .xo, $(KERNEL)))

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
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps -O3 -j 18 -D FPGABUILD

EXECUTABLE = driver
CMD_ARGS = $(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin

EMCONFIG_DIR = $(XCLBIN)/$(DSA)

BINARY_CONTAINERS += $(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin
BINARY_CONTAINER_bitstream_OBJS += $(XCLBIN)/bitstream.$(TARGET).$(DSA).xo

# Select the memory ports
CLFLAGS += --max_memory_ports $(KERNEL)
ifeq ($(USERAM), yes)
	LDCLFLAGS+=--sp $(KERNEL)_1.a:DDR[0] --sp $(KERNEL)_1.b:DDR[0] --sp $(KERNEL)_1.c:DDR[0]
	LDCLFLAGS+=--sp $(KERNEL)_2.a:DDR[1] --sp $(KERNEL)_2.b:DDR[1] --sp $(KERNEL)_2.c:DDR[1]
else
	LDCLFLAGS+=--sp $(KERNEL)_1.a:HBM[0] --sp $(KERNEL)_1.b:HBM[1] --sp $(KERNEL)_1.c:HBM[2]
endif


##########################

#Generates profile summary report
ifeq ($(PROFILE), yes)
LDCLFLAGS += --profile_kernel data:all:all:all
endif

#Generates debug summary report
ifeq ($(DEBUG), yes)
CLFLAGS += --dk protocol:all:all:all
endif


.PHONY: xilinx clean cleanall emconfig exe
exe: $(EXECUTABLE)
xilinx: $(exe) $(BINARY_CONTAINERS) emconfig


build: $(BINARY_CONTAINERS)

########################################################## MORE XILINX SPECIFIC STUFF
# Building kernel
$(XCLBIN)/$(TARGET)_%.xo: $(KERNEL_SRC)
	@echo ""
	@echo "> > > Compiling opencl source"
	@mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_bitstream) -c -k $* -I'$(<D)' -o'$@' '$<'

$(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin: $(KERNEL_BINS)
	@echo ""
	@echo "> > > Building binary kernel for $(TARGET)"
	@mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_bitstream) -l $(LDCLFLAGS) --nk $(KERNEL):$(CUNITS) -o'$@' $(+)

# $(XCLBIN)/bitstream.$(TARGET).$(DSA).xo: kernel.cl
# 	mkdir -p $(XCLBIN)
# 	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_bitstream) -c -k events_kernel -I'$(<D)' -o'$@' '$<'
# $(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin: $(BINARY_CONTAINER_bitstream_OBJS)
# 	mkdir -p $(XCLBIN)
# 	$(XOCC) $(CLFLAGS) --temp_dir $(BUILD_DIR_bitstream) -l $(LDCLFLAGS) --nk events_kernel:1 -o'$@' $(+)

# Building Host
$(EXECUTABLE): $(HOST_SRCS) $(HOST_HDRS)
	$(XXX) $(CXXFLAGS) $(HOST_SRCS) $(HOST_HDRS) -o '$@' $(LDFLAGS) -D FPGABUILD

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

ifeq ($(TARGET), hw)
EMULATION_MODE=""
else
EMULATION_MODE=$(TARGET)
endif

run: xilinx driver
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	cp -rf $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(EMULATION_MODE) ./$(EXECUTABLE) $(KERNEL) $(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin $(EVENTS) 1
else
	./$(EXECUTABLE) $(KERNEL) $(XCLBIN)/bitstream.$(TARGET).$(DSA).xclbin $(EVENTS) 1
endif
# 	sdx_analyze profile -i profile_summary.csv -f html
#################################################################################

$(TARGET_CPP): $(HOST_SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: run-gpu run-python run-cpu

run-cpu: $(TARGET_CPP)
	./$(TARGET_CPP) $(KERNEL) $(KERNEL_SRC) $(EVENTS) 2

run-gpu: $(TARGET_CPP)
	./$(TARGET_CPP) $(KERNEL) $(KERNEL_SRC) $(EVENTS) 0

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

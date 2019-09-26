CXX = g++
CF = -O3 -lOpenCL

EVENTS=1000000
DIMENSIONS=7

TARGET= cpp-opencl
SRC = host.cc

.PHONY: all clean run-cpp run-python

all: $(TARGET)

$(TARGET): $(SRC) 
	$(CXX) $(CF) $^ -o $@

run-cpp: $(TARGET)
	./$(TARGET) $(EVENTS) $(DIMENSIONS)

run-python:
	python vegas_mc_pyopencl.py -n $(EVENTS) -d $(DIMENSIONS)

clean:
	rm -f $(TARGET)

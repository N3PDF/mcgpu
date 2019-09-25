NC = nvcc
NF = -O3

EVENTS=1000000
DIMENSIONS=7

TARGET = cpp-cuda
SRC = cpp-cuda.cu kernel.cu

.PHONY: all clean run-cpp run-python

all: $(TARGET)

$(TARGET): $(SRC)
	$(NC) $(NF) $^ -o $@

run-cpp: $(TARGET)
	./$(TARGET) $(EVENTS) $(DIMENSIONS)

run-python:
	python vegas_mc_pycuda.py -n $(EVENTS) -d $(DIMENSIONS)

clean:
	rm -f cpp-cuda 

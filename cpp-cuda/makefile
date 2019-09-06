NC = nvcc
NF = -O3

TARGET = cpp-cuda
SRC = cpp-cuda.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NC) $(NF) $^ -o $@

run: $(TARGET)
	./$(TARGET)


clean:
	cpp-cuda

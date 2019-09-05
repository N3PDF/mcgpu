NC = nvcc

TARGET = cpp-cuda
SRC = cpp-cuda.cu

.PHONY: all  clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NC) $^ -o $@

run: $(TARGET)
	./$(TARGET)


clean:
	cpp-cuda

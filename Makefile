CPP=g++
NVC=nvcc
CFLAGS=-O2 -Wall
NVCFLAGS=-arch=compute_12 -O3 --use_fast_math
LIBS=

all: wave_cpu.x wave_cuda.x pml_cpu.x pml_cuda.x

run_cpu:
	./wave_cpu.x
	python plot.py

pml_cpu.x: main_pml.cpp cpu/cpu_pml_wave_2d.o
	$(NVC) $(NVCFLAGS) main_pml.cpp -o pml_cpu.x cpu/cpu_pml_wave_2d.o

pml_cuda.x: main_pml.cpp cuda/cuda_pml_wave_2d.o
	$(NVC) $(NVCFLAGS) -D USE_CUDA main_pml.cpp -o pml_cuda.x cuda/cuda_pml_wave_2d.o

wave_cpu.x: main.cpp cpu/cpu_wave_2d.o
	$(NVC) $(NVCFLAGS) main.cpp -o wave_cpu.x cpu/cpu_wave_2d.o

wave_cuda.x: main.cpp cuda/cuda_wave_2d.o
	$(NVC) $(NVCFLAGS) -D USE_CUDA main.cpp -o wave_cuda.x cuda/cuda_wave_2d.o


cpu/cpu_pml_wave_2d.o: cpu/cpu_pml_wave_2d.cpp cpu/cpu_pml_wave_2d_math.cpp
	$(NVC) $(NVCFLAGS) -c cpu/cpu_pml_wave_2d.cpp -o cpu/cpu_pml_wave_2d.o

cuda/cuda_pml_wave_2d.o: cuda/cuda_pml_wave_2d.cu cuda/cuda_pml_wave_2d_kernel.cu cuda/cuda_pml_wave_2d_kernel_math.cu
	$(NVC) $(NVCFLAGS) -c cuda/cuda_pml_wave_2d.cu -o cuda/cuda_pml_wave_2d.o

cpu/cpu_wave_2d.o: cpu/cpu_wave_2d.cpp
	$(NVC) $(NVCFLAGS) -c cpu/cpu_wave_2d.cpp -o cpu/cpu_wave_2d.o

cuda/cuda_wave_2d.o: cuda/cuda_wave_2d.cu cuda/cuda_wave_2d_kernel.cu
	$(NVC) $(NVCFLAGS) -c cuda/cuda_wave_2d.cu -o cuda/cuda_wave_2d.o

clean:
	rm *.x *.o cuda/*.x cuda/*.o cpu/*.x cpu/*.o

clean_frames:
	rm frames/frame*
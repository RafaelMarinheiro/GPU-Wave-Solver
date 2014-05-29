CPP=g++
NVC=nvcc
CFLAGS=-O2 -Wall
NVCFLAGS=-arch=compute_13
LIBS=

all: wave_cpu.x wave_cuda.x

run_cpu:
	./wave_cpu.x
	python plot.py

wave_cpu.x: main.cpp wave_solver.o
	$(NVC) $(NVCFLAGS) main.cpp -o wave_cpu.x wave_solver.o

wave_cuda.x: main.cpp cuda/cuda_wave_2d.o
	$(NVC) $(NVCFLAGS) -D USE_CUDA main.cpp -o wave_cuda.x cuda/cuda_wave_2d.o

wave_solver.o: wave_solver.h wave_solver.cpp
	$(NVC) $(NVCFLAGS) -c wave_solver.cpp -o wave_solver.o

cuda/cuda_wave_2d.o: cuda/cuda_wave_2d.cu cuda/cuda_wave_2d_kernel.cu
	$(NVC) $(NVCFLAGS) -c cuda/cuda_wave_2d.cu -o cuda/cuda_wave_2d.o

clean:
	rm *.x *.o cuda/*.x cuda/*.o

clean_frames:
	rm frames/frame*
CPP=g++
CFLAGS=-O2 -Wall
LIBS=

all: wave_cpu.x

run_cpu:
	./wave_cpu.x
	python plot.py

wave_cpu.x: main.cpp wave_solver.o
	$(CPP) $(CFLAGS) main.cpp -o wave_cpu.x wave_solver.o

wave_solver.o: wave_solver.h wave_solver.cpp
	$(CPP) $(CFLAGS) -c wave_solver.cpp -o wave_solver.o

clean:
	rm *.x *.o

clean_frames:
	rm frames/frame*
#include <cstdio>
#include <cmath>

#include "wave_solver.h"

Number_t gaussian(double x, double y, void * ctx){
	Number_t stddev = 0.01;
	Number_t mean = 0.5;
	Number_t var2 = stddev*stddev*2;
	Number_t term = sqrt((x-mean)*(x-mean) + (y-mean)*(y-mean));
	return stddev*exp(-term*term/var2)/sqrt(acos(-1)*var2);
}

void writeToFile(FILE * fp, Number_t * u, int nx, int ny){
	for(int i = 1; i <= ny; i++){
		for(int j =1; j <= nx; j++){
			fprintf(fp, "%.5lf ", u[j + i*(nx+2)]);
		}
		fprintf(fp, "\n");
	}
}

int main(int argc, char** argv){
	Wave_2d_t wave;

	int nx = 100;
	char filename[1024];
	int ny = nx;
	int nsteps = 200;
	Number_t c = 0.34029;
	Number_t * u = NULL;
	wave = wave_sim_init(0, 0, 1, 1,
						c, 0.5/(nx*c),
						nx, ny,
						gaussian,
						NULL);


	for(int step = 0; step < nsteps; step++){
		u = wave_sim_get_u(wave, 0);
		printf("Frame %d\n", step);
		sprintf(filename, "frames/frame%d", step);
		FILE *fp = fopen(filename, "w+");
		writeToFile(fp, u, nx, ny);
		fclose(fp);
		wave_sim_step(wave);
	}
}
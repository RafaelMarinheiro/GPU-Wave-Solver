#include "cuda_pml_wave_2d_kernel_math.cu"

__device__ Number_t w_get_pos(int j, int nn, Number_t vmin, Number_t vmax, Number_t dd){
	return ((j*vmax + (nn - j)*vmin)/nn) + dd/2;
}

__constant__ Number_t kernel_constants[11];

__global__ void cuda_pml_wave_2d_velocity_kernel(Number_t * u,
									const int nx, const int ny){
	#define BDIMX 16
	#define BDIMY 16

	__shared__ Number_t cache[BDIMX + 2][BDIMY + 2][3];

	Number_t c = kernel_constants[0];
	Number_t dt = kernel_constants[1];
	Number_t dx = kernel_constants[2];
	Number_t dy = kernel_constants[3];
	Number_t xmin = kernel_constants[4];
	Number_t xmax = kernel_constants[5];
	Number_t ymin = kernel_constants[6];
	Number_t ymax = kernel_constants[7];
	Number_t pml_strength = kernel_constants[8];
	Number_t pml_width = kernel_constants[9];
	Number_t density = kernel_constants[10];

	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if(j < nx && i < ny){

		int idx = 3*(j + nx*i);

		int bdy = min(BDIMY, ny-i);
		int bdx = min(BDIMX, nx-j);

		//Position in the cache
		int tx = threadIdx.x + 1;
		int ty = threadIdx.y + 1;

		//Fill the cache
		//cache[x][y][dim]
		if(threadIdx.y < 1){
			if(i != 0){
				int bidx = 3*(j + nx*(i-1));
				cache[tx][threadIdx.y][0] = u[bidx];
				cache[tx][threadIdx.y][1] = u[bidx + 1];
				cache[tx][threadIdx.y][2] = u[bidx + 2];
			} else{
				cache[tx][threadIdx.y][0] = 0;
				cache[tx][threadIdx.y][1] = 0;
				cache[tx][threadIdx.y][2] = 0;
			}
			if(i+bdy < ny){
				int bidx = 3*(j + (i+bdy)*nx);
				cache[tx][threadIdx.y+bdy+1][0] = u[bidx];
				cache[tx][threadIdx.y+bdy+1][1] = u[bidx + 1];
				cache[tx][threadIdx.y+bdy+1][2] = u[bidx + 2];
			} else{
				cache[tx][threadIdx.y+bdy+1][0] = 0;
				cache[tx][threadIdx.y+bdy+1][1] = 0;
				cache[tx][threadIdx.y+bdy+1][2] = 0;
			}
		}
		if(threadIdx.x < 1){
			if(j != 0){
				int bidx = 3*((j-1) + nx*i);
				cache[threadIdx.x][ty][0] = u[bidx];
				cache[threadIdx.x][ty][1] = u[bidx + 1];
				cache[threadIdx.x][ty][2] = u[bidx + 2];
			} else{
				cache[threadIdx.x][ty][0] = 0;
				cache[threadIdx.x][ty][1] = 0;
				cache[threadIdx.x][ty][2] = 0;
			}
			if(j+bdx < nx){
				int bidx = 3*((j+bdx) + i*nx);
				cache[threadIdx.x+bdx+1][ty][0] = u[bidx];
				cache[threadIdx.x+bdx+1][ty][1] = u[bidx + 1];
				cache[threadIdx.x+bdx+1][ty][2] = u[bidx + 2];
			} else{
				cache[threadIdx.x+bdx+1][ty][0] = 0;
				cache[threadIdx.x+bdx+1][ty][1] = 0;
				cache[threadIdx.x+bdx+1][ty][2] = 0;
			}
		}
		cache[tx][ty][0] = u[idx];
		cache[tx][ty][1] = u[idx+1];
		cache[tx][ty][2] = u[idx+2];
		__syncthreads();

		//Set the position
		Number_t bx = w_get_pos(j, nx, xmin, xmax, dx);
		Number_t by = w_get_pos(i, ny, ymin, ymax, dy);

		//Update velocities
		{
			Number_t oldVx = cache[tx][ty][1];
			Number_t oldVy = cache[tx][ty][2];
			Number_t newVx = 0;
			Number_t newVy = 0;

			Number_t absortion;
			Number_t update;
			Number_t gradient;
			//X
			if(j != nx-1){
				absortion = pml_wave_2d_absortion(bx+dx/2, xmin, xmax, pml_strength, pml_width);
				update = pml_wave_2d_vel_update(dt, absortion);
				gradient = pml_wave_2d_gradient(dt, absortion, dx, density);

				newVx = oldVx*update + gradient*(cache[tx+1][ty][0]-cache[tx][ty][0]);
			}

			//Y
			if(i != ny-1){
				absortion = pml_wave_2d_absortion(by+dy/2, ymin, ymax, pml_strength, pml_width);
				update = pml_wave_2d_vel_update(dt, absortion);
				gradient = pml_wave_2d_gradient(dt, absortion, dy, density);

				newVy = oldVy*update + gradient*(cache[tx][ty+1][0]-cache[tx][ty][0]);
			}

			cache[tx][ty][1] = newVx;
			cache[tx][ty][2] = newVy;
		}

		u[idx+1] = cache[tx][ty][1];
		u[idx+2] = cache[tx][ty][2];
	}
}

__global__ void cuda_pml_wave_2d_pressure_kernel(Number_t * u,
									const int nx, const int ny){
	#define BDIMX 16
	#define BDIMY 16

	__shared__ Number_t cache[BDIMX + 2][BDIMY + 2][3];

	Number_t c = kernel_constants[0];
	Number_t dt = kernel_constants[1];
	Number_t dx = kernel_constants[2];
	Number_t dy = kernel_constants[3];
	Number_t xmin = kernel_constants[4];
	Number_t xmax = kernel_constants[5];
	Number_t ymin = kernel_constants[6];
	Number_t ymax = kernel_constants[7];
	Number_t pml_strength = kernel_constants[8];
	Number_t pml_width = kernel_constants[9];
	Number_t density = kernel_constants[10];

	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if(j < nx && i < ny){

		int idx = 3*(j + nx*i);

		int bdy = min(BDIMY, ny-i);
		int bdx = min(BDIMX, nx-j);

		//Position in the cache
		int tx = threadIdx.x + 1;
		int ty = threadIdx.y + 1;

		//Fill the cache
		//cache[x][y][dim]
		if(threadIdx.y < 1){
			if(i != 0){
				int bidx = 3*(j + nx*(i-1));
				cache[tx][threadIdx.y][0] = u[bidx];
				cache[tx][threadIdx.y][1] = u[bidx + 1];
				cache[tx][threadIdx.y][2] = u[bidx + 2];
			} else{
				cache[tx][threadIdx.y][0] = 0;
				cache[tx][threadIdx.y][1] = 0;
				cache[tx][threadIdx.y][2] = 0;
			}
			if(i+bdy < ny){
				int bidx = 3*(j + (i+bdy)*nx);
				cache[tx][threadIdx.y+bdy+1][0] = u[bidx];
				cache[tx][threadIdx.y+bdy+1][1] = u[bidx + 1];
				cache[tx][threadIdx.y+bdy+1][2] = u[bidx + 2];
			} else{
				cache[tx][threadIdx.y+bdy+1][0] = 0;
				cache[tx][threadIdx.y+bdy+1][1] = 0;
				cache[tx][threadIdx.y+bdy+1][2] = 0;
			}
		}
		if(threadIdx.x < 1){
			if(j != 0){
				int bidx = 3*((j-1) + nx*i);
				cache[threadIdx.x][ty][0] = u[bidx];
				cache[threadIdx.x][ty][1] = u[bidx + 1];
				cache[threadIdx.x][ty][2] = u[bidx + 2];
			} else{
				cache[threadIdx.x][ty][0] = 0;
				cache[threadIdx.x][ty][1] = 0;
				cache[threadIdx.x][ty][2] = 0;
			}
			if(j+bdx < nx){
				int bidx = 3*((j+bdx) + i*nx);
				cache[threadIdx.x+bdx+1][ty][0] = u[bidx];
				cache[threadIdx.x+bdx+1][ty][1] = u[bidx + 1];
				cache[threadIdx.x+bdx+1][ty][2] = u[bidx + 2];
			} else{
				cache[threadIdx.x+bdx+1][ty][0] = 0;
				cache[threadIdx.x+bdx+1][ty][1] = 0;
				cache[threadIdx.x+bdx+1][ty][2] = 0;
			}
		}
		cache[tx][ty][0] = u[idx];
		cache[tx][ty][1] = u[idx+1];
		cache[tx][ty][2] = u[idx+2];
		__syncthreads();

		//Set the position
		Number_t bx = w_get_pos(j, nx, xmin, xmax, dx);
		Number_t by = w_get_pos(i, ny, ymin, ymax, dy);

		//Update pressure
		{

			Number_t abs_x = pml_wave_2d_absortion(bx+dx/2, xmin, xmax, pml_strength, pml_width);
			Number_t abs_y = pml_wave_2d_absortion(by+dy/2, ymin, ymax, pml_strength, pml_width);
			Number_t dir_x = pml_wave_2d_directional(dt, abs_x);
			Number_t dir_y = pml_wave_2d_directional(dt, abs_y);
			Number_t upd_x = pml_wave_2d_pre_update(dt, abs_x, dir_x);
			Number_t upd_y = pml_wave_2d_pre_update(dt, abs_y, dir_y);
			Number_t div_x = pml_wave_2d_pre_divergence(density, c, dir_x, dx);
			Number_t div_y = pml_wave_2d_pre_divergence(density, c, dir_y, dy);

			Number_t oldU = cache[tx][ty][0];
			Number_t newU = oldU*(upd_x + upd_y)/2
						  + div_x*(cache[tx][ty][1]-cache[tx-1][ty][1])
						  + div_y*(cache[tx][ty][2]-cache[tx][ty-1][2]);
						  ;

			cache[tx][ty][0] = newU;
		}

		//Write back to the global memory
		// u[idx] = threadIdx.x;
		u[ idx ] = cache[tx][ty][0];
	}
}
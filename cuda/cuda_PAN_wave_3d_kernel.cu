__constant__ Number_t kernel_constants[18];
 
#include "cuda_PAN_wave_3d_kernel_math.cu"

__device__ __forceinline__ Number_t w_get_pos(int j, int nn, Number_t vmin, Number_t vmax, Number_t dd){
	return ((j*vmax + (nn - j)*vmin)/nn) + dd/2;
}


#define RRADX 16
#define RRADY 16

__global__ void cuda_pan_wave_3d_velocity_kernel(Number_t * __restrict__ u,
												 const Number_t * __restrict__ grad,
												 const bool * __restrict__ isBulk,
												 Number_t t,
												 const int nx,
												 const int ny,
												 const int nz,
												 int field){
	const Number_t dt = kernel_constants[1];
	const Number_t idt = 1/dt;
	const Number_t dx = kernel_constants[2];
	const Number_t dy = kernel_constants[3];
	const Number_t xmin = kernel_constants[4];
	const Number_t xmax = kernel_constants[5];
	const Number_t ymin = kernel_constants[6];
	const Number_t ymax = kernel_constants[7];
	const Number_t zmin = kernel_constants[8];
	const Number_t zmax = kernel_constants[9];
	const Number_t pml_strength = kernel_constants[10];
	const Number_t pml_width = kernel_constants[11];
	const Number_t density = kernel_constants[12];
	const Number_t timepulse = kernel_constants[16];
	const Number_t dz = kernel_constants[17];
	const Number_t mit = PAN_Mitchelli(t, timepulse);


	Number_t local_z[4];
	Number_t local_old[4];

	bool ibulk;
	bool bulk_z;

	__shared__ Number_t cache[RRADX+2][RRADY+2];
	__shared__ bool cache_bulk[RRADX+2][RRADY+2];

	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int j = blockIdx.y*blockDim.y + threadIdx.y;

	const int bdx = min(RRADX, nx-i);
	const int bdy = min(RRADY, ny-j);

	if(i < nx && j < ny){
		const Number_t bx = w_get_pos(i, nx, xmin, xmax, dx);
		const Number_t by = w_get_pos(j, ny, ymin, ymax, dy);

		const int tx = threadIdx.x + 1;
		const int ty = threadIdx.y + 1;

		const Number_t absortion_x = pan_wave_3d_absortion(bx+dx/2, xmin, xmax, pml_strength, pml_width);
		const Number_t update_x = pan_wave_3d_vel_update(idt, absortion_x);
		const Number_t gradient_x = pan_wave_3d_gradient(idt, absortion_x, dx, density);
		
		const Number_t absortion_y = pan_wave_3d_absortion(by+dy/2, ymin, ymax, pml_strength, pml_width);
		const Number_t update_y = pan_wave_3d_vel_update(idt, absortion_y);
		const Number_t gradient_y = pan_wave_3d_gradient(idt, absortion_y, dy, density);
		

		//Compute the first guy:
		{
			const int bbbase = 4*(i + nx*j);

			local_z[0] = u[bbbase+0];
			local_z[1] = u[bbbase+1];
			local_z[2] = u[bbbase+2];
			local_z[3] = u[bbbase+3];
			bulk_z = isBulk[(i + nx*j)];
		}

		#pragma unroll 2
		for(int other = 0; other < nz-1; other++){
			const int k = other;
			const int base = i + nx*(j + ny*k);
			const int idx = 4*base;
			const Number_t bz = w_get_pos(k, nz, zmin, zmax, dz);

			Number_t local_new[4];

			local_old[0] = local_z[0];
			local_old[1] = local_z[1];
			local_old[2] = local_z[2];
			local_old[3] = local_z[3];
			local_new[0] = local_old[0];
			local_new[1] = local_new[2] = local_new[3] = 0;

			ibulk = bulk_z;

			cache[tx][ty] = local_old[0];
			cache_bulk[tx][ty] = ibulk;

			__syncthreads();

			if(threadIdx.x == 0){
				if(i+bdx < nx){
					const int base = 4*((i+bdx) + nx*(j + ny*k));
					cache[tx+bdx][ty] = u[base+0];
					cache_bulk[tx+bdx][ty] = isBulk[(i+bdx) + nx*(j + ny*k)];
				} else{
					cache[tx+bdx][ty] = 0;
					cache_bulk[tx+bdx][ty] = false;
				}
			}
			if(threadIdx.y == 0){
				if(j+bdy < ny){
					const int base = 4*(i + nx*((j+bdy) + ny*k));
					cache[tx][ty+bdy] = u[base+0];
					cache_bulk[tx][ty+bdy] = isBulk[(i + nx*((j+bdy) + ny*k))];
				} else{
					cache[tx][ty+bdy] = 0;
					cache_bulk[tx][ty+bdy] = false;
				}
			}
			__syncthreads();

			//Solve for X
			if(i != nx-1){
				const bool otherbulk = cache_bulk[tx+1][ty];
				if(ibulk && otherbulk){
					local_new[1] = local_old[1]*update_x + gradient_x*(cache[tx+1][ty] - cache[tx][ty]);
				} else if(ibulk || otherbulk){
					Number_t gradi = grad[3*base]*mit;
					if(ibulk){
						gradi = -gradi;
					}
					local_new[1] = local_old[1] + gradi*PAN_boundary(bx+dx/2, by, bz, field, 0);
				}
			}
			//Solve for Y
			if(j != ny-1){
				const bool otherbulk = cache_bulk[tx][ty+1];
				if(ibulk && otherbulk){
					local_new[2] = local_old[2]*update_y + gradient_y*(cache[tx][ty+1] - cache[tx][ty]);
				} else if(ibulk || otherbulk){
					Number_t gradi = grad[3*base+1]*mit;
					if(ibulk){
						gradi = -gradi;
					}
					local_new[2] = local_old[2] + gradi*PAN_boundary(bx, by+dy/2, bz, field, 1);
				}
			}


			//Solve for Z
			{
				const int bbase = (i + nx*(j + ny*(k+1)));
				const int bidx = 4*bbase;
				local_z[0] = u[bidx+0];
				local_z[1] = u[bidx+1];
				local_z[2] = u[bidx+2];
				local_z[3] = u[bidx+3];
				bulk_z = isBulk[bbase];
				const bool otherbulk = bulk_z;
				if(ibulk && otherbulk){
					const Number_t absortion = pan_wave_3d_absortion(bz+dz/2, zmin, zmax, pml_strength, pml_width);
					const Number_t update = pan_wave_3d_vel_update(idt, absortion);
					const Number_t gradient = pan_wave_3d_gradient(idt, absortion, dz, density);
					local_new[3] = local_old[3]*update + gradient*(local_z[0] - cache[tx][ty]);
				} else if(ibulk || otherbulk){
					Number_t gradi = grad[3*base+2]*mit;
					if(ibulk){
						gradi = -gradi;
					}
					local_new[3] = local_old[3] + gradi*PAN_boundary(bx, by, bz+dz/2, field, 2);
				}
			}
			//u[idx + 0] = local_new[0];
			u[idx + 1] = local_new[1];
			u[idx + 2] = local_new[2];
			u[idx + 3] = local_new[3];
		}
	}
}


__global__ void cuda_pan_wave_3d_pressure_kernel(Number_t * u,
												 bool * isBulk,
												 const int nx,
												 const int ny,
												 const int nz){

	const Number_t c = kernel_constants[0];
	const Number_t dt = kernel_constants[1];
	const Number_t idt = 1/dt;
	const Number_t dx = kernel_constants[2];
	const Number_t dy = kernel_constants[3];
	const Number_t xmin = kernel_constants[4];
	const Number_t xmax = kernel_constants[5];
	const Number_t ymin = kernel_constants[6];
	const Number_t ymax = kernel_constants[7];
	const Number_t zmin = kernel_constants[8];
	const Number_t zmax = kernel_constants[9];
	const Number_t pml_strength = kernel_constants[10];
	const Number_t pml_width = kernel_constants[11];
	const Number_t density = kernel_constants[12];
	const Number_t dz = kernel_constants[17];

	Number_t local_z;

	__shared__ Number_t cache[RRADX+2][RRADY+2][4];

	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int j = blockIdx.y*blockDim.y + threadIdx.y;

	const int bdx = min(RRADX, nx-i);
	const int bdy = min(RRADY, ny-j);

	if(i < nx && j < ny){
		const Number_t bx = w_get_pos(i, nx, xmin, xmax, dx);
		const Number_t abs_x = pan_wave_3d_absortion(bx+dx/2, xmin, xmax, pml_strength, pml_width);
		const Number_t dir_x = pan_wave_3d_directional(idt, abs_x);
		const Number_t upd_x = pan_wave_3d_pre_update(idt, abs_x, dir_x);
		const Number_t div_x = pan_wave_3d_pre_divergence(density, c, dir_x, dx);

		const Number_t by = w_get_pos(j, ny, ymin, ymax, dy);
		const Number_t abs_y = pan_wave_3d_absortion(by+dy/2, ymin, ymax, pml_strength, pml_width);
		const Number_t dir_y = pan_wave_3d_directional(idt, abs_y);
		const Number_t upd_y = pan_wave_3d_pre_update(idt, abs_y, dir_y);
		const Number_t div_y = pan_wave_3d_pre_divergence(density, c, dir_y, dy);

		const int tx = threadIdx.x + 1;
		const int ty = threadIdx.y + 1;

		{
			const int bbase = 4*(i + nx*j);
			cache[tx][ty][0] = u[bbase+0];
			cache[tx][ty][1] = u[bbase+1];
			cache[tx][ty][2] = u[bbase+2];
			cache[tx][ty][3] = u[bbase+3];
		}

		for(int other = 1; other < nz; other++){
			const int k = other;
			const int base = i + nx*(j+ny*k);
			const int idx = 4*base;

			{
				local_z = cache[tx][ty][3];
				cache[tx][ty][0] = u[idx+0];
				cache[tx][ty][1] = u[idx+1];
				cache[tx][ty][2] = u[idx+2];
				cache[tx][ty][3] = u[idx+3];
			}
			if(threadIdx.x == 0){
				if(i != 0){
					const int base = 4*((i-1) + nx*(j + ny*k));
					cache[0][ty][0] = u[base+0];
					cache[0][ty][1] = u[base+1];
					cache[0][ty][2] = u[base+2];
					cache[0][ty][3] = u[base+3];
				} else{
					cache[0][ty][0] = 0;
					cache[0][ty][1] = 0;
					cache[0][ty][2] = 0;
					cache[0][ty][3] = 0;
				}
			}
			if(threadIdx.y == 0){
				if(j != 0){
					const int base = 4*(i + nx*((j-1) + ny*k));
					cache[tx][0][0] = u[base+0];
					cache[tx][0][1] = u[base+1];
					cache[tx][0][2] = u[base+2];
					cache[tx][0][3] = u[base+3];
				} else{
					cache[tx][0][0] = 0;
					cache[tx][0][1] = 0;
					cache[tx][0][2] = 0;
					cache[tx][0][3] = 0;
				}
			}

			__syncthreads();
			const bool ibulk = isBulk[base];

			if(ibulk){
				Number_t update = 0;
				Number_t diver = 0;

				//Solve for X
				{
					diver += div_x*(cache[tx][ty][1]-cache[tx-1][ty][1]);
				}

				//Solve for Y
				{
					diver += div_y*(cache[tx][ty][2]-cache[tx][ty-1][2]);
				}

				//Solve for Z
				{
					const Number_t bz = w_get_pos(k, nz, zmin, zmax, dz);
					const Number_t abs_d = pan_wave_3d_absortion(bz+dz/2, zmin, zmax, pml_strength, pml_width);
					const Number_t dir_d = pan_wave_3d_directional(idt, abs_d);
					const Number_t upd_d = pan_wave_3d_pre_update(idt, abs_d, dir_d);
					const Number_t div_d = pan_wave_3d_pre_divergence(density, c, dir_d, dz);

					update += upd_d/3;
					diver += div_d*(cache[tx][ty][3]-local_z);
					update = (upd_x + upd_y + upd_d)/3;
				}
				const Number_t local_new = update*cache[tx][ty][0] + diver;

				u[idx+0] = local_new;
			}
		}
	}
}

__global__ void cuda_pan_wave_3d_listen_kernel(const Number_t * __restrict__ u,
											   Number_t * __restrict__ output,
										       const int num_listening,
											   const Number_t * __restrict__ listening_positions,
											   const int nx,
											   const int ny,
											   const int nz){

	const Number_t dx = kernel_constants[2];
	const Number_t dy = kernel_constants[3];
	const Number_t xmin = kernel_constants[4];
	const Number_t xmax = kernel_constants[5];
	const Number_t ymin = kernel_constants[6];
	const Number_t ymax = kernel_constants[7];
	const Number_t zmin = kernel_constants[8];
	const Number_t zmax = kernel_constants[9];
	const Number_t dz = kernel_constants[17];

	int ii = blockIdx.x*blockDim.x + threadIdx.x;

	if(ii < num_listening){
		Number_t data[2][2][2];
		Number_t x = listening_positions[3*ii];
		Number_t y = listening_positions[3*ii + 1];
		Number_t z = listening_positions[3*ii + 2];

		const int i = (int) floor((x - dx/2 - xmin)/dx);
		const int j = (int) floor((y - dy/2 - ymin)/dy);
		const int k = (int) floor((z - dz/2 - zmin)/dz);

		x = (x - w_get_pos(i, nx, xmin, xmax, dx))/dx;
		y = (y - w_get_pos(j, ny, ymin, ymax, dy))/dy;
		z = (z - w_get_pos(k, nz, zmin, zmax, dz))/dz;
	
		data[0][0][0] = u[4*(i + nx*(j + ny*k))];
		data[0][0][1] = u[4*((i+1) + nx*(j + ny*k))];
		data[0][1][0] = u[4*(i + nx*((j+1) + ny*k))];
		data[0][1][1] = u[4*((i+1) + nx*((j+1) + ny*k))];
		data[1][0][0] = u[4*(i + nx*(j + ny*(k+1)))];
		data[1][0][1] = u[4*((i+1) + nx*(j + ny*(k+1)))];
		data[1][1][0] = u[4*(i + nx*((j+1) + ny*(k+1)))];
		data[1][1][1] = u[4*((i+1) + nx*((j+1) + ny*(k+1)))];

		data[0][0][0] = (1-x)*data[0][0][0] + x*data[0][0][1];
		data[0][1][0] = (1-x)*data[0][1][0] + x*data[0][1][1];
		data[1][0][0] = (1-x)*data[1][0][0] + x*data[1][0][1];
		data[1][1][0] = (1-x)*data[1][1][0] + x*data[1][1][1];

		data[0][0][0] = (1-y)*data[0][0][0] + x*data[0][1][0];
		data[1][0][0] = (1-y)*data[1][0][0] + x*data[1][1][0];

		data[0][0][0] = (1-z)*data[0][0][0] + z*data[1][0][0];

		output[ii] = data[0][0][0];
	}
}

#undef RRADX
#undef RRADY	
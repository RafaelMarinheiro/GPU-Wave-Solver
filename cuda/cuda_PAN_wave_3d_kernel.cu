__constant__ Number_t kernel_constants[18];

#include "cuda_PAN_wave_3d_kernel_math.cu"

__device__ __forceinline__ Number_t w_get_pos(int j, int nn, Number_t vmin, Number_t vmax, Number_t dd){
	return ((j*vmax + (nn - j)*vmin)/nn) + dd/2;
}

__global__ void cuda_pan_wave_3d_velocity_kernel(Number_t * __restrict__ u,
												 const Number_t * __restrict__ grad,
												 const bool * __restrict__ isBulk,
												 Number_t t,
												 const int nx,
												 const int ny,
												 const int nz){
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
	Number_t local_new[4];

	bool ibulk;
	bool bulk_z;

	__shared__ Number_t cache[18][18];
	__shared__ bool cache_bulk[18][18];

	const int bdx = 16;
	const int bdy = 16;

	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < nx && j < ny){
		const Number_t bx = w_get_pos(i, nx, xmin, xmax, dx);
		const Number_t by = w_get_pos(j, ny, ymin, ymax, dy);

		const int tx = threadIdx.x + 1;
		const int ty = threadIdx.y + 1;

		//Compute the first guy:
		const int bbbase = 4*(i + nx*j);

		local_z[0] = u[bbbase+0];
		local_z[1] = u[bbbase+1];
		local_z[2] = u[bbbase+2];
		local_z[3] = u[bbbase+3];
		bulk_z = isBulk[(i + nx*j)];

		#pragma unroll
		for(int other = 0; other < nz; other++){
			int k = other;
			int base = i + nx*(j + ny*k);
			int idx = 4*base;
			const Number_t bz = w_get_pos(k, nz, zmin, zmax, dz);

			local_old[0] = local_z[0];
			local_old[1] = local_z[1];
			local_old[2] = local_z[2];
			local_old[3] = local_z[3];
			local_new[0] = local_old[0];
			local_new[1] = local_new[2] = local_new[3] = 0;
			ibulk = bulk_z;

			cache[tx][ty] = local_old[0];
			cache_bulk[tx][ty] = ibulk;

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
					const Number_t absortion = pan_wave_3d_absortion(bx+dx/2, xmin, xmax, pml_strength, pml_width);
					const Number_t update = pan_wave_3d_vel_update(idt, absortion);
					const Number_t gradient = pan_wave_3d_gradient(idt, absortion, dx, density);
					local_new[1] = local_old[1]*update + gradient*(cache[tx+1][ty] - cache[tx][ty]);
				} else if(ibulk || otherbulk){
					Number_t gradi = grad[3*base]*mit;
					if(ibulk){
						gradi = -gradi;
					}
					local_new[1] = local_old[1] + gradi*PAN_boundary(bx+dx/2, by, bz, 0, 0);
				}
			}
			//Solve for Y
			if(j != ny-1){
				const bool otherbulk = cache_bulk[tx][ty+1];
				if(ibulk && otherbulk){
					const Number_t absortion = pan_wave_3d_absortion(by+dy/2, ymin, ymax, pml_strength, pml_width);
					const Number_t update = pan_wave_3d_vel_update(idt, absortion);
					const Number_t gradient = pan_wave_3d_gradient(idt, absortion, dy, density);
					local_new[2] = local_old[2]*update + gradient*(cache[tx][ty+1] - cache[tx][ty]);
				} else if(ibulk || otherbulk){
					Number_t gradi = grad[3*base+1]*mit;
					if(ibulk){
						gradi = -gradi;
					}
					local_new[2] = local_old[2] + gradi*PAN_boundary(bx, by+dy/2, bz, 0, 1);
				}
			}

			//Solve for Z
			if(k != nz-1){
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
					local_new[3] = local_old[3] + gradi*PAN_boundary(bx, by, bz+dz/2, 0, 2);
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

	Number_t local_me[6];
	Number_t local_new[6];
	Number_t local_v_me[6];
	Number_t local_v_other[6];

	Number_t c = kernel_constants[0];
	Number_t dt = kernel_constants[1];
	Number_t idt = 1/dt;
	Number_t dx = kernel_constants[2];
	Number_t dy = kernel_constants[3];
	Number_t xmin = kernel_constants[4];
	Number_t xmax = kernel_constants[5];
	Number_t ymin = kernel_constants[6];
	Number_t ymax = kernel_constants[7];
	Number_t zmin = kernel_constants[8];
	Number_t zmax = kernel_constants[9];
	Number_t pml_strength = kernel_constants[10];
	Number_t pml_width = kernel_constants[11];
	Number_t density = kernel_constants[12];
	Number_t dz = kernel_constants[17];

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < nx && j < ny){
		for(int other = 0; other < nz; other++){
			int k = other;
			int base = i + nx*(j+ny*k);
			if(isBulk[base]){
				int idx = 4*6*base;
				Number_t update = 0;
				Number_t local_div[6] = {0, 0, 0, 0, 0, 0};
				Number_t abs_d;
				Number_t dir_d;
				Number_t upd_d;
				Number_t div_d;
				local_me[0] = u[idx+0];
				local_me[1] = u[idx+1];
				local_me[2] = u[idx+2];
				local_me[3] = u[idx+3];
				local_me[4] = u[idx+4];
				local_me[5] = u[idx+5];

				//Solve for X
				{
					local_v_me[0] = u[idx+6];
					local_v_me[1] = u[idx+7];
					local_v_me[2] = u[idx+8];
					local_v_me[3] = u[idx+9];
					local_v_me[4] = u[idx+10];
					local_v_me[5] = u[idx+11];
					if(i != 0){
						int bbase = (i-1) + nx*(j+ny*k);
						int bidx = 4*6*bbase + 6;
						local_v_other[0] = u[bidx + 0];
						local_v_other[1] = u[bidx + 1];
						local_v_other[2] = u[bidx + 2];
						local_v_other[3] = u[bidx + 3];
						local_v_other[4] = u[bidx + 4];
						local_v_other[5] = u[bidx + 5];
					} else{
						local_v_other[0] = 0;
						local_v_other[1] = 0;
						local_v_other[2] = 0;
						local_v_other[3] = 0;
						local_v_other[4] = 0;
						local_v_other[5] = 0;
					}

					Number_t bx = w_get_pos(i, nx, xmin, xmax, dx);
					abs_d = pan_wave_3d_absortion(bx+dx/2, xmin, xmax, pml_strength, pml_width);
					dir_d = pan_wave_3d_directional(idt, abs_d);
					upd_d = pan_wave_3d_pre_update(idt, abs_d, dir_d);
					div_d = pan_wave_3d_pre_divergence(density, c, dir_d, dx);

					update += upd_d/3;
					local_div[0] += div_d*(local_v_me[0] - local_v_other[0]);
					local_div[1] += div_d*(local_v_me[1] - local_v_other[1]);
					local_div[2] += div_d*(local_v_me[2] - local_v_other[2]);
					local_div[3] += div_d*(local_v_me[3] - local_v_other[3]);
					local_div[4] += div_d*(local_v_me[4] - local_v_other[4]);
					local_div[5] += div_d*(local_v_me[5] - local_v_other[5]);
				}

				//Solve for Y
				{
					local_v_me[0] = u[idx+12];
					local_v_me[1] = u[idx+13];
					local_v_me[2] = u[idx+14];
					local_v_me[3] = u[idx+15];
					local_v_me[4] = u[idx+16];
					local_v_me[5] = u[idx+17];
					if(j != 0){
						int bbase = i + nx*((j-1)+ny*k);
						int bidx = 4*6*bbase + 12;
						local_v_other[0] = u[bidx + 0];
						local_v_other[1] = u[bidx + 1];
						local_v_other[2] = u[bidx + 2];
						local_v_other[3] = u[bidx + 3];
						local_v_other[4] = u[bidx + 4];
						local_v_other[5] = u[bidx + 5];
					} else{
						local_v_other[0] = 0;
						local_v_other[1] = 0;
						local_v_other[2] = 0;
						local_v_other[3] = 0;
						local_v_other[4] = 0;
						local_v_other[5] = 0;
					}

					Number_t by = w_get_pos(j, ny, ymin, ymax, dy);
					abs_d = pan_wave_3d_absortion(by+dy/2, ymin, ymax, pml_strength, pml_width);
					dir_d = pan_wave_3d_directional(idt, abs_d);
					upd_d = pan_wave_3d_pre_update(idt, abs_d, dir_d);
					div_d = pan_wave_3d_pre_divergence(density, c, dir_d, dy);

					update += upd_d/3;
					local_div[0] += div_d*(local_v_me[0] - local_v_other[0]);
					local_div[1] += div_d*(local_v_me[1] - local_v_other[1]);
					local_div[2] += div_d*(local_v_me[2] - local_v_other[2]);
					local_div[3] += div_d*(local_v_me[3] - local_v_other[3]);
					local_div[4] += div_d*(local_v_me[4] - local_v_other[4]);
					local_div[5] += div_d*(local_v_me[5] - local_v_other[5]);
				}

				//Solve for Z
				{
					local_v_me[0] = u[idx+18];
					local_v_me[1] = u[idx+19];
					local_v_me[2] = u[idx+20];
					local_v_me[3] = u[idx+21];
					local_v_me[4] = u[idx+22];
					local_v_me[5] = u[idx+23];
					if(k != 0){
						int bbase = i + nx*(j+ny*(k-1));
						int bidx = 4*6*bbase + 18;
						local_v_other[0] = u[bidx + 0];
						local_v_other[1] = u[bidx + 1];
						local_v_other[2] = u[bidx + 2];
						local_v_other[3] = u[bidx + 3];
						local_v_other[4] = u[bidx + 4];
						local_v_other[5] = u[bidx + 5];
					} else{
						local_v_other[0] = 0;
						local_v_other[1] = 0;
						local_v_other[2] = 0;
						local_v_other[3] = 0;
						local_v_other[4] = 0;
						local_v_other[5] = 0;
					}

					Number_t bz = w_get_pos(k, nz, zmin, zmax, dz);
					abs_d = pan_wave_3d_absortion(bz+dz/2, zmin, zmax, pml_strength, pml_width);
					dir_d = pan_wave_3d_directional(idt, abs_d);
					upd_d = pan_wave_3d_pre_update(idt, abs_d, dir_d);
					div_d = pan_wave_3d_pre_divergence(density, c, dir_d, dz);

					update += upd_d/3;
					local_div[0] += div_d*(local_v_me[0] - local_v_other[0]);
					local_div[1] += div_d*(local_v_me[1] - local_v_other[1]);
					local_div[2] += div_d*(local_v_me[2] - local_v_other[2]);
					local_div[3] += div_d*(local_v_me[3] - local_v_other[3]);
					local_div[4] += div_d*(local_v_me[4] - local_v_other[4]);
					local_div[5] += div_d*(local_v_me[5] - local_v_other[5]);
				}
				local_new[0] = update*local_me[0] + local_div[0];
				local_new[1] = update*local_me[1] + local_div[1];
				local_new[2] = update*local_me[2] + local_div[2];
				local_new[3] = update*local_me[3] + local_div[3];
				local_new[4] = update*local_me[4] + local_div[4];
				local_new[5] = update*local_me[5] + local_div[5];

				u[idx+0] = local_new[0];
				u[idx+1] = local_new[1];
				u[idx+2] = local_new[2];
				u[idx+3] = local_new[3];	
				u[idx+4] = local_new[4];
				u[idx+5] = local_new[5];
			}
		}
	}
}
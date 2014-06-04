#ifndef Cuda_PAN_3D_WAVE_SOLVER_H
#define Cuda_PAN_3D_WAVE_SOLVER_H

#include "../defines.h" 
#include <boost/function.hpp>

struct Cuda_PAN_Wave_3d_sim_data_t;

typedef Cuda_PAN_Wave_3d_sim_data_t * Cuda_PAN_Wave_3d_t;

Cuda_PAN_Wave_3d_t wave_sim_init(Number_t xmin, Number_t ymin, Number_t zmin,
								 Number_t xmax, Number_t ymax, Number_t zmax,
								 Number_t c, Number_t dt,
								 Number_t cellsize,
								 int listening_count,
								 Number_t * listening_positions,
								 const Wave_InitialCondition3D & initial,
								 const Wave_BoundaryEvaluator3D & boundary,
								 Number_t xcenter, Number_t ycenter, Number_t zcenter,
								 const Wave_GradientEvaluator3D & gradient,
								 Number_t pml_width,
								 Number_t pml_strength,
								 Number_t pulse);

void wave_sim_free(Cuda_PAN_Wave_3d_t wave);

Number_t wave_sim_get_x(Cuda_PAN_Wave_3d_t wave, int j);
Number_t wave_sim_get_y(Cuda_PAN_Wave_3d_t wave, int i);
Number_t wave_sim_get_z(Cuda_PAN_Wave_3d_t wave, int k);

Number_t * wave_sim_get_u(Cuda_PAN_Wave_3d_t wave);

void wave_sim_step(Cuda_PAN_Wave_3d_t wave);
void wave_sim_apply_boundary(Cuda_PAN_Wave_3d_t wave);

//positions-> 3*Number_t*pos array
//output->6*Number_t*pos array
void wave_listen_at(Cuda_PAN_Wave_3d_t wave, Number_t * positions, Number_t * output);

#endif
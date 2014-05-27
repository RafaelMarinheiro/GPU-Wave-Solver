#ifndef WAVE_SOLVER_H
#define WAVE_SOLVER_H

#include "defines.h"

struct Wave_2d_sim_data_t;

typedef Wave_2d_sim_data_t * Wave_2d_t;

/**
 * @brief Initializes the simulation data
 * 
 * @param xmin Minimum X of the domain
 * @param ymin Minimum Y of the domain
 * @param xmax Maximum X of the domain
 * @param ymax Maximum Y of the domain
 * @param c Constant of the Wave Equation
 * @param dt Time Step
 * @param dx Grid Cell Size
 * @param init_function Function to initialize the wave
 * @param ctx Context passed to init_function
 * @return A struct that describes the simulation data
 */
Wave_2d_t wave_sim_init(Number_t xmin, Number_t ymin,
						Number_t xmax, Number_t ymax,
						Number_t c, Number_t dt,
						int nx, int ny,
						Number_t (*init_function)(Number_t, Number_t, void *),
						void * ctx);

void wave_sim_free(Wave_2d_t wave);

Number_t wave_sim_get_x(Wave_2d_t wave, int j);
Number_t wave_sim_get_y(Wave_2d_t wave, int i);

//Acessing element:
// u[j + i*(nx+2)]
// i -> y, j->x

Number_t * wave_sim_get_u(Wave_2d_t wave, int offset);

void wave_sim_step(Wave_2d_t wave);
void wave_sim_apply_boundary(Wave_2d_t wave);

#endif 
/* WAVE_SOLVER_H */
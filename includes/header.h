#ifndef HEADER_H
# define HEADER_H

# include <easy_sdl.h>

typedef struct			s_data
{
	t_esdl				*esdl;
	SDL_Surface			*surf;
	SDL_Texture			*tex;
	int					fractal;
}						t_data;

typedef enum			e_fractal
{
	MANDELBROT, JULIA, BURNING_SHIP
}						t_fractal;

void					mandelbrot(t_data *data);
void					julia(t_data *data);
void					burning_ship(t_data *data);

#endif
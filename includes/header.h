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
	MANDELBROT, JULIA, BURNING_SHIP, TRICORN, SIZE = 4
}						t_fractal;

void					mandelbrot(t_data *data);
void					julia(t_data *data);
void					burning_ship(t_data *data);
void					tricorn(t_data *data);

int						check_arg(int argc, char **argv, t_data *data);
int						show_usage(void);


#endif
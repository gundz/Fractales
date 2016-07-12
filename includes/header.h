#ifndef HEADER_H
#define HEADER_H

#include <easy_sdl.h>

typedef struct			s_data
{
	t_esdl				*esdl;
	SDL_Texture			*screen;
}						t_data;

void	mandelbrot(t_data *data, SDL_Surface *surf);
#endif
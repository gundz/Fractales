#ifndef HEADER_H
#define HEADER_H

#include <easy_sdl.h>

typedef struct			s_data
{
	t_esdl				*esdl;
	SDL_Surface			*surf;
	SDL_Texture			*tex;
}						t_data;

void	mandelbrot(t_data *data);
#endif
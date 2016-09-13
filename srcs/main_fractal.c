#include <header.h>

int					check_input(t_data *data)
{
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		return (1);
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		return (1);
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_LEFT) == 1)
		return (1);
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_RIGHT) == 1)
		return (1);
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_UP) == 1)
		return (1);
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_DOWN) == 1)
		return (1);
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_KP_PLUS) == 1)
		return (1);
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_KP_MINUS) == 1)
		return (1);
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_Q) == 1)
		return (1);
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_W) == 1)
		return (1);
	return (0);
}

void				change_fractal(t_data *data)
{
	if (data->esdl->en.in.key[SDL_SCANCODE_Q] == 1)
	{
		data->fractal--;
		data->esdl->en.in.key[SDL_SCANCODE_Q] = 0;
	}
	else if (data->esdl->en.in.key[SDL_SCANCODE_W] == 1)
	{
		data->fractal++;
		data->esdl->en.in.key[SDL_SCANCODE_W] = 0;
	}
	if (data->fractal >= SIZE)
		data->fractal = 0;
	else if (data->fractal < 0)
		data->fractal = SIZE;
}

void				main_fractal(t_data *data)
{
	if (data->fractal == MANDELBROT)
		mandelbrot(data);
	if (data->fractal == JULIA)
		julia(data);
	if (data->fractal == BURNING_SHIP)
		burning_ship(data);
	if (data->fractal == TRICORN)
		tricorn(data);
	data->tex = SDL_CreateTextureFromSurface(data->esdl->en.ren, data->surf);
	SDL_RenderClear(data->esdl->en.ren);
	SDL_RenderCopy(data->esdl->en.ren, data->tex, NULL, NULL);
	SDL_RenderPresent(data->esdl->en.ren);
	SDL_DestroyTexture(data->tex);
}
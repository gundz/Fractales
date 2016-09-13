#include <easy_sdl.h>
#include <libft.h>
#include <header.h>
#include <sys/time.h>

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

	printf("%d %d\n", data->fractal, SIZE);
}

void				test(t_data *data)
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

void				init(t_data *data)
{
	data->surf = Esdl_create_surface(SDL_RX, SDL_RY);
}

void				quit(t_data *data)
{
	SDL_FreeSurface(data->surf);
}

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

int					main(int argc, char **argv)
{
	t_data			data;
	t_esdl			esdl;

	data.esdl = &esdl;
	if (check_arg(argc, argv, &data) == -1)
		return (show_usage());
	if (Esdl_init(&esdl, 640, 480, 120, "Engine") == -1)
		return (-1);
	init(&data);
	test(&data);
	while (esdl.run)
	{
		Esdl_update_events(&esdl.en.in, &esdl.run);
		change_fractal(&data);

		if (check_input(&data) || data.fractal == JULIA)
			test(&data);
		Esdl_fps_limit(&esdl);
		Esdl_fps_counter(&esdl);
	}
	quit(&data);
	Esdl_quit(&esdl);
	return (0);
}

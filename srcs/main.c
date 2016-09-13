#include <easy_sdl.h>
#include <libft.h>
#include <header.h>

void				init(t_data *data)
{
	data->surf = Esdl_create_surface(SDL_RX, SDL_RY);
}

void				quit(t_data *data)
{
	SDL_FreeSurface(data->surf);
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
	main_fractal(&data);
	while (esdl.run)
	{
		change_fractal(&data);
		Esdl_update_events(&esdl.en.in, &esdl.run);
		if (check_input(&data) || data.fractal == JULIA)
			main_fractal(&data);
		Esdl_fps_limit(&esdl);
		Esdl_fps_counter(&esdl);
	}
	quit(&data);
	Esdl_quit(&esdl);
	return (0);
}

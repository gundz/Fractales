#include <easy_sdl.h>
#include <header.h>
#include <sys/time.h>


void					test(t_data *data)
{
	//mandelbrot(data);
	julia(data);

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

int					main(int argc, char **argv)
{
	t_data			data;
	t_esdl			esdl;

	data.esdl = &esdl;

	if (Esdl_init(&esdl, 640, 480, 120, "Engine") == -1)
		return (-1);
	init(&data);
	while (esdl.run)
	{
		Esdl_update_events(&esdl.en.in, &esdl.run);

		test(&data);

		Esdl_fps_limit(&esdl);
		Esdl_fps_counter(&esdl);
	}
	quit(&data);
	Esdl_quit(&esdl);

	(void)argc;
	(void)argv;
	return (0);
}
#include <easy_sdl.h>
#include <header.h>
#include <sys/time.h>


void					test(t_data *data)
{
	SDL_Surface			*surf = Esdl_create_surface(SDL_RX, SDL_RY);
	SDL_Texture			*tex;

	struct timeval stop, start;
	gettimeofday(&start, NULL);
	
	mandelbrot(data, surf);

	gettimeofday(&stop, NULL);
	printf("took %fs\n", (double)(stop.tv_usec - start.tv_usec) / 1000000000.0);


	tex = SDL_CreateTextureFromSurface(data->esdl->en.ren, surf);
	SDL_FreeSurface(surf);
	SDL_RenderClear(data->esdl->en.ren);
	SDL_RenderCopy(data->esdl->en.ren, tex, NULL, NULL);
	SDL_RenderPresent(data->esdl->en.ren);
	SDL_DestroyTexture(tex);
}

void				init(t_data *data)
{
	//data->square = SDL_CreateTextureFromSurface(data->esdl->en.ren, surf);
	//SDL_FreeSurface(surf);
	(void)data;
}

void				quit(t_data *data)
{
	//SDL_DestroyTexture(data->square);
	(void)data;
}

int					main(int argc, char **argv)
{
	t_data			data;
	t_esdl			esdl;

	data.esdl = &esdl;

	if (Esdl_init(&esdl, 1024, 768, 120, "Engine") == -1)
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
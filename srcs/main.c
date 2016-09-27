/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:36:17 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:36:17 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <easy_sdl.h>
#include <libft.h>
#include <header.h>

void				init(t_data *data)
{
	data->surf = esdl_create_surface(SDL_RX, SDL_RY);
}

void				quit(t_data *data)
{
	SDL_FreeSurface(data->surf);
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

int					main(int argc, char **argv)
{
	t_data			data;
	t_esdl			esdl;

	data.esdl = &esdl;
	if (check_arg(argc, argv, &data) == -1)
		return (show_usage());
	if (esdl_init(&esdl, 640, 480, "Fractol") == -1)
		return (-1);
	init(&data);
	main_fractal(&data);
	while (esdl.run)
	{
		esdl_update_events(&esdl.en.in, &esdl.run);
		if (check_input(&data) > 0 || data.fractal == JULIA)
			main_fractal(&data);
		esdl_fps_limit(&esdl);
		esdl_fps_counter(&esdl);
	}
	quit(&data);
	esdl_quit(&esdl);
	return (0);
}

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
		if (check_input(&data) > 0 || change_fractal(&data) > 0 \
			|| data.fractal == JULIA)
			main_fractal(&data);
		esdl_fps_limit(&esdl);
		esdl_fps_counter(&esdl);
	}
	quit(&data);
	esdl_quit(&esdl);
	return (0);
}

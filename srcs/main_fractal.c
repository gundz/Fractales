/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main_fractal.c                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:36:27 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:36:28 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <header.h>
#include <libft.h>

int				change_fractal(t_data *data)
{
	int			ret;

	ret = 0;
	if (data->esdl->en.in.key[SDL_SCANCODE_Q] == 1)
	{
		data->fractal--;
		data->esdl->en.in.key[SDL_SCANCODE_Q] = 0;
		ret++;
	}
	else if (data->esdl->en.in.key[SDL_SCANCODE_W] == 1)
	{
		data->fractal++;
		data->esdl->en.in.key[SDL_SCANCODE_W] = 0;
		ret++;
	}
	if (data->fractal >= SIZE)
		data->fractal = 0;
	else if (data->fractal < 0)
		data->fractal = SIZE;
	return (ret);
}

void			main_fractal(t_data *data)
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

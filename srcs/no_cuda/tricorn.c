/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   tricorn.c                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:30:16 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:30:17 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <header.h>
#include <tricorn.h>

void					tricorn_kernel(t_data *data, t_fractal fractal, int x, int y)
{
	double		pr, pi;
	double		zx, zy;
	double		zx2, zy2;
	int			i;

	pr = fractal.cx + (x - SDL_RX / 2) * fractal.zoom + fractal.movex;
	pi = fractal.cy + (y - SDL_RY / 2) * fractal.zoom + fractal.movey;
	zx = 0;
	zy = 0;
	i = 0;
	while (i < fractal.maxiteration)
	{
		zx2 = zx * zx;
		zy2 = zy * zy;
		zy = -(2 * zx * zy) + pi;
		zx = zx2 - zy2 + pr;
		if (zx2 + zy2 > 4)
			break ;
		i++;
	}
	int brightness = color_it(zx2, zy2, i, 100);
	esdl_put_pixel(data->surf, x, y, esdl_hsv_to_rgb(brightness % 256, 255, 255 * (i < fractal.maxiteration)));
}

void					tricorn(t_data *data)
{
	int					x;
	int					y;
	static t_fractal	fractal = {(2.5 / 480), 0, 0, 50, 0, 0, 0, 0, 0, 0};

	fractal.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	fractal.my = data->esdl->en.in.m_y - SDL_RY / 2;
	fractal_input(data, &fractal);
	y = 0;
	while (y < SDL_RY)
	{
		x = 0;
		while (x < SDL_RX)
		{
			tricorn_kernel(data, fractal, x, y);
			x++;
		}
		y++;
	}
}

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   julia.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:28:59 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:29:00 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <header.h>

void					julia_kernel(t_data *data, t_fractal fractal, int x, int y)
{
	double				pr, pi;
	double				zx, zy;
	double				zx2, zy2;
	int					i;

	pr = 0.003 * fractal.mx;
	pi = 0.003 * fractal.my;
	zx = fractal.cx + (x - SDL_RX / 2) * fractal.zoom + fractal.movex;
	zy = fractal.cy + (y - SDL_RX / 2) * fractal.zoom + fractal.movey;
	i = 0;
	while (i < fractal.maxiteration)
	{
		zx2 = zx * zx;
		zy2 = zy * zy;
		zy = 2 * zx * zy + pi;
		zx = zx2 - zy2 + pr;
		if (zx2 + zy2 > 4)
			break ;
		i++;
	}
	if (i == fractal.maxiteration)
		esdl_put_pixel(data->surf, x, y, 0xFFFFFFFF);
	else
		esdl_put_pixel(data->surf, x, y, (int)(i * 255 / 400) << 24 | (i % 255)  << 16 | 255 << 8 | 255);
}

void					julia(t_data *data)
{
	static t_fractal	fractal = {(2.5 / 480), 0, 0, 50, 0, 0.5, 0, 0, 0, 0};
	int					x;
	int					y;

	fractal.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	fractal.my = data->esdl->en.in.m_y - SDL_RY / 2;
	fractal_input(data, &fractal);
	y = 0;
	while (y < SDL_RY)
	{
		x = 0;
		while (x < SDL_RX)
		{
			julia_kernel(data, fractal, x, y);
			x++;
		}
		y++;
	}
}

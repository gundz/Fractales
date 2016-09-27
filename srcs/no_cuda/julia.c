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
#include <julia.h>

void			julia_kernel(t_data *data, t_fractal fractal, int x, int y)
{
	double		pr, pi;
	double		zx, zy;
	double		zx2, zy2;
	int			i;

	pr = 0.002 * fractal.mx;
	pi = 0.002 * fractal.my;
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
		esdl_put_pixel(data->surf, x, y, 0);
}

void				caca(t_data *data, t_fractal *fractal)
{
	fractal->oldcx = fractal->cx;
	fractal->oldcy = fractal->cy;

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		fractal->movex -= 0.0001 / fractal->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		fractal->movex += 0.0001 / fractal->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		fractal->movey -= 0.0001 / fractal->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		fractal->movey += 0.0001 / fractal->zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		fractal->zoom = fractal->zoom / 1.05;
		fractal->cx = (fractal->oldcx) + (fractal->mx * 0.05) * fractal->zoom;
		fractal->cy = (fractal->oldcy) + (fractal->my * 0.05) * fractal->zoom;
		fractal->maxiteration *= 1.0025;
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		fractal->zoom = fractal->zoom * 1.05;
		fractal->cx = (fractal->oldcx) + (fractal->mx * 0.05) * fractal->zoom;
		fractal->cy = (fractal->oldcy) + (fractal->my * 0.05) * fractal->zoom;
		fractal->maxiteration *= 0.9975;
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
		fractal->maxiteration *= 1.1;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
		fractal->maxiteration *= 0.9;
}

void				julia(t_data *data)
{
	static t_fractal fractal = {(2.5 / 480), 0, 0, 50, 0, 0.5, 0, 0, 0, 0};
	int				x;
	int				y;

	fractal.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	fractal.my = data->esdl->en.in.m_y - SDL_RY / 2;

	caca(data, &fractal);

	//fractal_input(data, &fractal);
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

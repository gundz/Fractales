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

int					julia_color(double new_re, double new_im, int i, int max_iteration)
{
	double			z;
	int				brightness;

	z = sqrt(new_re * new_re + new_im * new_im);
	brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
	return (brightness << 24 | (i % 255) << 16 | 255 << 8 | 255);
}

void				julia_kernel(t_data *data, t_julia julia, int x, int y)
{
	double			pr;
	double			pi;
	double			new_re;
	double			new_im;
	double			old_re;
	double			old_im;
	int				i;

	pr = 0.001 * julia.mx;
	pi = 0.001 * julia.my;
	new_re = (x - SDL_RX / 2) / (0.5 * julia.zoom * SDL_RX) + julia.movex;
	new_im = (y - SDL_RY / 2) / (0.5 * julia.zoom * SDL_RY) + julia.movey;
	i = 0;
	while (((new_re * new_re + new_im * new_im) < 4) && i < julia.maxiteration)
	{
		old_re = new_re;
		old_im = new_im;
		new_re = old_re * old_re - old_im * old_im + pr;
		new_im = 2 * old_re * old_im + pi;
		i++;
	}
	esdl_put_pixel(data->surf, x, y, julia_color(new_re, new_im, i, julia.maxiteration));
}

void
julia_input(t_data *data, t_julia *julia)
{
	julia->oldcx = julia->cx;
	julia->oldcy = julia->cy;

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		julia->movex -= 0.0001 / julia->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		julia->movex += 0.0001 / julia->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		julia->movey -= 0.0001 / julia->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		julia->movey += 0.0001 / julia->zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		julia->zoom = julia->zoom / 1.05;
		julia->cx = (julia->oldcx) + (julia->mx * 0.05) * julia->zoom;
		julia->cy = (julia->oldcy) + (julia->my * 0.05) * julia->zoom;
		julia->maxiteration *= 1.0025;
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		julia->zoom = julia->zoom * 1.05;
		julia->cx = (julia->oldcx) + (julia->mx * 0.05) * julia->zoom;
		julia->cy = (julia->oldcy) + (julia->my * 0.05) * julia->zoom;
		julia->maxiteration *= 0.9975;
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
		julia->maxiteration *= 1.1;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
		julia->maxiteration *= 0.9;
}

void				julia(t_data *data)
{
	static t_julia julia = {(2.5 / 480), 0, 0, 400, 0, 0, 0, 0, 0, 0};
	int				x;
	int				y;

	julia.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	julia.my = data->esdl->en.in.m_y - SDL_RY / 2;
	julia_input(data, &julia);
	y = 0;
	while (y < SDL_RY)
	{
		x = 0;
		while (x < SDL_RX)
		{
			julia_kernel(data, julia, x, y);
			x++;
		}
		y++;
	}
}

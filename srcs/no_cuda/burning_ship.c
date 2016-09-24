/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   burning_ship.c                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:29:21 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:29:22 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <header.h>
#include <burning_ship.h>

int						burning_ship_color(double new_re, double new_im, int i, int max_iteration)
{
	double				z;
	int					brightness;

	z = sqrt(new_re * new_re + new_im * new_im);
	brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
	return (brightness << 24 | (i % 255) << 16 | 255 << 8 | 255);
}

void					burning_ship_kernel(t_data *data, t_burning burning, int x, int y)
{
	double				pr;
	double				pi;
	double				new_re;
	double				new_im;
	double				old_re;
	double				old_im;
	int					i;

	pr = (x - SDL_RX / 2) / (0.5 * burning.zoom * SDL_RX) + burning.movex;
	pi = (y - SDL_RY / 2) / (0.5 * burning.zoom * SDL_RY) + burning.movey;
	new_re = new_im = old_re = old_im = 0;
	i = 0;
	while (((new_re * new_re + new_im * new_im) < 4) && i < burning.maxiteration)
	{
		new_re = (old_re * old_re - old_im * old_im) + pr;
		new_im = (fabs(old_re * old_im) * 2) + pi;
		old_re = new_re;
		old_im = new_im;
		i++;
	}
	esdl_put_pixel(data->surf, x, y, burning_ship_color(new_re, new_im, i, burning.maxiteration));
}

void
burning_input(t_data *data, t_burning *burning)
{
	burning->oldcx = burning->cx;
	burning->oldcy = burning->cy;

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		burning->movex -= 0.0001 / burning->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		burning->movex += 0.0001 / burning->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		burning->movey -= 0.0001 / burning->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		burning->movey += 0.0001 / burning->zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		burning->zoom = burning->zoom / 1.05;
		burning->cx = (burning->oldcx) + (burning->mx * 0.05) * burning->zoom;
		burning->cy = (burning->oldcy) + (burning->my * 0.05) * burning->zoom;
		burning->maxiteration *= 1.0025;
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		burning->zoom = burning->zoom * 1.05;
		burning->cx = (burning->oldcx) + (burning->mx * 0.05) * burning->zoom;
		burning->cy = (burning->oldcy) + (burning->my * 0.05) * burning->zoom;
		burning->maxiteration *= 0.9975;
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
		burning->maxiteration *= 1.1;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
		burning->maxiteration *= 0.9;
}

void					burning_ship(t_data *data)
{
	int					x;
	int					y;
	static t_burning burning = {(2.5 / 480), 0, 0, 400, 0, 0, 0, 0, 0, 0};

	burning.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	burning.my = data->esdl->en.in.m_y - SDL_RY / 2;
	burning_input(data, &burning);
	y = 0;
	while (y < SDL_RY)
	{
		x = 0;
		while (x < SDL_RX)
		{
			burning_ship_kernel(data, burning, x, y);
			x++;
		}
		y++;
	}
}

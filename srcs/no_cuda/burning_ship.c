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

	pr = (x - SDL_RX / 2) / (0.5 * burning.zoom * SDL_RX) + burning.moveX;
	pi = (y - SDL_RY / 2) / (0.5 * burning.zoom * SDL_RY) + burning.moveY;
	new_re = new_im = old_re = old_im = 0;
	i = 0;
	while (((new_re * new_re + new_im * new_im) < 4) && i < burning.maxIteration)
	{
		new_re = (old_re * old_re - old_im * old_im) + pr;
		new_im = (fabs(old_re * old_im) * 2) + pi;
		old_re = new_re;
		old_im = new_im;
		i++;
	}
	Esdl_put_pixel(data->surf, x, y, burning_ship_color(new_re, new_im, i, burning.maxIteration));
}

void					burning_ship(t_data *data)
{
	int					x;
	int					y;
	static t_burning	burning = {1, 0, 0, 50};

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		burning.moveX -= 0.01 / burning.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		burning.moveX += 0.01 / burning.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		burning.moveY -= 0.01 / burning.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		burning.moveY += 0.01 / burning.zoom * 10;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		burning.zoom += 0.01 * burning.zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		burning.zoom -= 0.01 * burning.zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
	{
		burning.maxIteration *= 1.1;
		printf("Max iterations = %d\n", burning.maxIteration);
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
	{
		burning.maxIteration *= 0.9;
		printf("Max iterations = %d\n", burning.maxIteration);
	}
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

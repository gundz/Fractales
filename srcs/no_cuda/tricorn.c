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

int						tricorn_color(double new_re, double new_im, int i, int max_iteration)
{
	double				z;
	int					brightness;

	z = sqrt(new_re * new_re + new_im * new_im);
	brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
	return (brightness << 24 | (i % 255) << 16 | 255 << 8 | 255);
}

void					tricorn_kernel(t_data *data, t_tricorn tricorn, int x, int y)
{
	double				pr;
	double				pi;
	double				new_re;
	double				new_im;
	double				old_re;
	double				old_im;
	int					i;

	pr = (x - SDL_RX / 2) / (0.5 * tricorn.zoom * SDL_RX) + tricorn.moveX;
	pi = (y - SDL_RY / 2) / (0.5 * tricorn.zoom * SDL_RY) + tricorn.moveY;
	new_re = new_im = old_re = old_im = 0;
	i = 0;
	while (((new_re * new_re + new_im * new_im) < 4) && i < tricorn.maxIteration)
	{
		old_re = new_re;
		old_im = new_im;
		new_re = old_re * old_re - old_im * old_im + pr;
		new_im = -(2 * old_re * old_im + pi);
		i++;
	}
	esdl_put_pixel(data->surf, x, y, tricorn_color(new_re, new_im, i, tricorn.maxIteration));
}

void					tricorn(t_data *data)
{
	int					x;
	int					y;
	static t_tricorn	tricorn = {1, -0.5, 0, 100};

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		tricorn.moveX -= 0.01 / tricorn.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		tricorn.moveX += 0.01 / tricorn.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		tricorn.moveY -= 0.01 / tricorn.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		tricorn.moveY += 0.01 / tricorn.zoom * 10;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		tricorn.zoom += 0.01 * tricorn.zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		tricorn.zoom -= 0.01 * tricorn.zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
	{
		tricorn.maxIteration *= 1.1;
		printf("Max iterations = %d\n", tricorn.maxIteration);
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
	{
		tricorn.maxIteration *= 0.9;
		printf("Max iterations = %d\n", tricorn.maxIteration);
	}
	y = 0;
	while (y < SDL_RY)
	{
		x = 0;
		while (x < SDL_RX)
		{
			tricorn_kernel(data, tricorn, x, y);
			x++;
		}
		y++;
	}
}

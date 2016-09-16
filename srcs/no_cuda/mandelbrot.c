/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mandelbrot.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:28:52 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:28:54 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <header.h>
#include <mandelbrot.h>

int						mandelbrot_color(double new_re, double new_im, int i, int max_iteration)
{
	double				z;
	int					brightness;

	z = sqrt(new_re * new_re + new_im * new_im);
	brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
	return (brightness << 24 | (i % 255) << 16 | 255 << 8 | 255);
}

void					mandelbrot_kernel(t_data *data, t_mandelbrot mandelbrot, int x, int y)
{
	double				pr;
	double				pi;
	double				new_re;
	double				new_im;
	double				old_re;
	double				old_im;
	int					i;

	pr = (x - SDL_RX / 2) / (0.5 * mandelbrot.zoom * SDL_RX) + mandelbrot.moveX;
	pi = (y - SDL_RY / 2) / (0.5 * mandelbrot.zoom * SDL_RY) + mandelbrot.moveY;
	new_re = new_im = old_re = old_im = 0;
	i = 0;
	while (((new_re * new_re + new_im * new_im) < 4) && i < mandelbrot.maxIteration)
	{
		old_re = new_re;
		old_im = new_im;
		new_re = old_re * old_re - old_im * old_im + pr;
		new_im = 2 * old_re * old_im + pi;
		i++;
	}
	Esdl_put_pixel(data->surf, x, y, mandelbrot_color(new_re, new_im, i, mandelbrot.maxIteration));
}

void					mandelbrot(t_data *data)
{
	int					x;
	int					y;
	static t_mandelbrot	mandelbrot = {1, -0.5, 0, 100};

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		mandelbrot.moveX -= 0.01 / mandelbrot.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		mandelbrot.moveX += 0.01 / mandelbrot.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		mandelbrot.moveY -= 0.01 / mandelbrot.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		mandelbrot.moveY += 0.01 / mandelbrot.zoom * 10;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		mandelbrot.zoom += 0.01 * mandelbrot.zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		mandelbrot.zoom -= 0.01 * mandelbrot.zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
	{
		mandelbrot.maxIteration *= 1.1;
		printf("Max iterations = %d\n", mandelbrot.maxIteration);
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
	{
		mandelbrot.maxIteration *= 0.9;
		printf("Max iterations = %d\n", mandelbrot.maxIteration);
	}
	y = 0;
	while (y < SDL_RY)
	{
		x = 0;
		while (x < SDL_RX)
		{
			mandelbrot_kernel(data, mandelbrot, x, y);
			x++;
		}
		y++;
	}
}

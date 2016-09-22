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
	double		pr, pi;
	double		zx, zy;
	double		zx2, zy2;
	int			i;

	pr = mandelbrot.cx + (x - SDL_RX / 2) * mandelbrot.zoom + mandelbrot.movex;
	pi = mandelbrot.cy + (y - SDL_RY / 2) * mandelbrot.zoom + mandelbrot.movey;
	zx = 0;
	zy = 0;
	i = 0;
	while (i < mandelbrot.maxiteration)
	{
		zx2 = zx * zx;
		zy2 = zy * zy;
		zy = 2 * zx * zy + pi;
		zx = zx2 - zy2 + pr;
		if (zx2 + zy2 > 4)
			break ;
		i++;
	}
	if (i == mandelbrot.maxiteration)
		esdl_put_pixel(data->surf, x, y, 0xFFFFFFFF);
	else
		esdl_put_pixel(data->surf, x, y, i * 255 / mandelbrot.caca << 24 | (i % 255)  << 16 | 255 << 8 | 255);
}

void					mandelbrot_input(t_data *data, t_mandelbrot *mandelbrot)
{
	mandelbrot->oldcx = mandelbrot->cx;
	mandelbrot->oldcy = mandelbrot->cy;

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		mandelbrot->movex -= 0.0001 / mandelbrot->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		mandelbrot->movex += 0.0001 / mandelbrot->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		mandelbrot->movey -= 0.0001 / mandelbrot->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		mandelbrot->movey += 0.0001 / mandelbrot->zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		mandelbrot->cx = mandelbrot->oldcx + mandelbrot->mx * mandelbrot->zoom;
		mandelbrot->cy = mandelbrot->oldcy + mandelbrot->my * mandelbrot->zoom;

		mandelbrot->zoom = mandelbrot->zoom / 1.1;
		mandelbrot->maxiteration *= 1.0025;
		data->esdl->en.in.button[SDL_BUTTON_LEFT] = 0;
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		mandelbrot->cx = mandelbrot->oldcx + mandelbrot->mx * mandelbrot->zoom;
		mandelbrot->cy = mandelbrot->oldcy + mandelbrot->my * mandelbrot->zoom;

		mandelbrot->zoom = mandelbrot->zoom * 1.1;
		mandelbrot->maxiteration *= 0.9975;
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
		mandelbrot->maxiteration *= 1.1;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
		mandelbrot->maxiteration *= 0.9;
}

void					mandelbrot(t_data *data)
{
	int					x;
	int					y;
	static t_mandelbrot	mandelbrot = {(2.5 / 480), 0, 0, 200, 0, 0, 0, 0, 0, 0};
	mandelbrot.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	mandelbrot.my = data->esdl->en.in.m_y - SDL_RY / 2;

	mandelbrot_input(data, &mandelbrot);
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

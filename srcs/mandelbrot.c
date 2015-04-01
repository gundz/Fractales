/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mandelbrot.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/03/16 17:04:07 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 02:57:19 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <mlx_lib.h>
#include <thread.h>
#include <fractol.h>

void					init_mandelbrot(t_fract *data)
{
	data->nb_it = 800;
	data->pos.x = 0;
	data->pos.y = 0;
	data->m_pos.x = 0;
	data->m_pos.y = 0;
	data->pos_inc = 0.1;
	data->zoom.x = 1;
	data->zoom.y = 100;
	data->zoom_inc = 0.5;
}

inline static void		set_color(const int x, const int y,
	t_fract *data, t_tab *tab)
{
	//SDL_Color			scolor;

	if (data->i == data->nb_it)
		put_pixel(data->surf, x, y, 0xFFFFFF);
	else
	{	
		if (data->i % 2 == 0)
			put_pixel(data->surf, x, y, (((data->i / data->nb_it) << 8) + 0x00FF00));
		else if (data->i % 10 == 0)
			put_pixel(data->surf, x, y, (((data->i / data->nb_it) << 16) + 0xAABFCA));
		else
			put_pixel(data->surf, x, y, (((data->i / data->nb_it) << 16) + 0xAAFF00));

		// scolor.r = 0;
		// scolor.g = 0;
		// if (data->i % 2 == 0)
		// 	scolor.g = data->i * 255 / data->nb_it;
		// if (data->i % 3 == 0)
		// 	scolor.r = data->i * 255 / data->nb_it;
		// if (data->i % 5 == 0)
		// {
		// 	scolor.r = data->i * 255 / data->nb_it;
		// 	scolor.g = data->i * 127 / data->nb_it;
		// 	scolor.b = data->i * 255 / data->nb_it;
		// }
		// scolor.b = data->i * 255 / data->nb_it;
		// scolor.a = 255;
		// put_pixel(data->surf, x, y,((data->i) << 16) + sdl_color_to_int(scolor));

	}
	(void)tab;
}
#include <stdio.h>
void					*mandelbrot(void *arg, const int x, const int y)
{
	t_thread			*thread = (t_thread *)arg;
	t_v2d				c;
	t_v2d				z;
	t_v2d				z2;
	long double			tmp;
	t_fract				*data;

	data = thread->data;

	double x1 = -2.1 / data->zoom.x;
	double x2 = 0.6 / data->zoom.x;
	double y1 = -1.2 / data->zoom.x;
	double y2 = 1.2 / data->zoom.x;

	double zx = RX / (x2 - x1);
	double zy = RY / (y2 - y1);

	c.x = (x / zx) + x1 + data->pos.x;
	c.y = (y / zy) + y1 + data->pos.y;
	z.x = 0;
	z.y = 0;
	z2.x = 0;
	z2.y = 0;
	data->i = 0;
	while ((z2.x + z.y) < 100 && data->i < data->nb_it)
	{
		z2.x = z.x * z.x;
		z2.y = z.y * z.y;
		tmp = z.x;
		z.x = z2.x - z2.y + c.x;
		z.y = (tmp * z.y + tmp * z.y) + c.y;
		data->i++;
	}
	set_color(x, y, data, thread->tab);
	return (NULL);
}

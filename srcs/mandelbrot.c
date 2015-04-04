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
#include <math.h>

void					init_mandelbrot(t_fract *data)
{
	int					i;

	data->max_it = 300;
	data->zoomp.x = 0;
	data->zoomp.y = 0;
	data->zoom = 1;
	data->pos.x = 0;
	data->pos.y = 0;
	data->pos_inc = 0.1;
	i = 0;
	while (i < 256)
	{
		data->c_map[i] = rgb_to_uint(i, i, i);
		data->c_map[i + 1 * 256] = rgb_to_uint(0, 127, 255 - i);
		data->c_map[i + 2 * 256] = rgb_to_uint(i, i, 0);
		data->c_map[i + 3 * 256] = rgb_to_uint(0, 127 - i, 0);
		i++;
	}
}

inline static void		set_color(const int x, const int y,
	t_fract *data, t_v2d tmp)
{
	if (data->i == data->max_it)
	{
		put_pixel(data->surf, x, y, 0xFFFFFF);
	}
	else
	{
		data->i = data->i - log(log(tmp.x + tmp.y)) / log(2);
		data->i = ((NM_COLOR - 1) * data->i) / data->max_it;
		put_pixel(data->surf, x, y, data->c_map[(int)data->i]);
	}
}

void					*mandelbrot(void *arg, const int x, const int y)
{
	t_thread			*thread;
	t_fract				*data;
	t_v2d				coor;
	t_v2d				tmp;
	t_v2d				tmp2;

	thread = (t_thread *)arg;
	data = thread->data;
	set_v2d(0.0, 0.0, &coor);
	set_v2d(0.0, 0.0, &tmp2);
	data->i = 0;
	while ((tmp2.x + tmp2.y) < 4 && data->i < data->max_it)
	{
		tmp.x = coor.x;
		tmp.y = coor.y;
		tmp2.x = tmp.x * tmp.x;
		tmp2.y = tmp.y * tmp.y;
		coor.x = tmp2.x - tmp2.y + (((x + data->zoomp.x) \
			/ data->zoom / RX) * 4 - 2) + data->pos.x;
		coor.y = (tmp.x * tmp.y) + (tmp.x * tmp.y) + \
			(((y + data->zoomp.y) / data->zoom / RY) * 4 - 2) + data->pos.y;
		data->i++;
	}
	set_color(x, y, data, tmp2);
	return (NULL);
}

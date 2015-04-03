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

void					init_julia(t_fract *data)
{
	data->max_it = 150;
	data->zoomp.x = 0;
	data->zoomp.y = 0;
	data->zoom = 100;
	data->coor.x = 0.285;
	data->coor.y = 0.01;
}

inline static void		set_color(const int x, const int y,
	t_fract *data)
{
	if (data->i == data->max_it)
		put_pixel(data->surf, x, y, 0xFFFFFF);
	else
	{
		put_pixel(data->surf, x, y, (((data->i) << 8) + 0x00FF00));
	}
}

void					*julia(void *arg, const int x, const int y)
{
	t_thread			*thread;
	t_fract				*data;
	t_v2d				coor;
	t_v2d				tmp;
	t_v2d				tmp2;

	thread = (t_thread *)arg;
	data = thread->data;
	set_v2d(((x - RX2) / data->zoom), ((y - RY2) / data->zoom), &coor);
	set_v2d(0.0, 0.0, &tmp2);
	data->i = 0;
	while ((tmp2.x + tmp2.y) < 4 && data->i < data->max_it)
	{
		tmp.x = coor.x;
		tmp.y = coor.y;
		tmp2.x = tmp.x * tmp.x;
		tmp2.y = tmp.y * tmp.y;
		coor.x = tmp2.x - tmp2.y + data->coor.x;
		coor.y = (tmp.x * coor.y + tmp.x * coor.y) + data->coor.y;
		data->i++;
	}
	set_color(x, y, data);
	return (NULL);
}

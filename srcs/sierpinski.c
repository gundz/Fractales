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

void					init_sierpinski(t_fract *data)
{
	data->max_it = 10;
	data->zoomp.x = 0;
	data->zoomp.y = 0;
	data->zoom = 1;
}

void					*sierpinski(void *arg, const int x, const int y)
{
	t_thread			*thread;
	t_fract				*data;
	t_v2i				coor;

	thread = (t_thread *)arg;
	data = thread->data;
	coor = set_v2i(
		ABS(x + data->zoomp.x / data->zoom),
		ABS(y + data->zoomp.y / data->zoom),
		NULL);
	data->i = 0;
	while ((coor.x > 0 || coor.y > 0) && data->i < data->max_it)
	{
		if ((coor.x % 3) == 1 && (coor.y % 3) == 1)
		{
			put_pixel(data->surf, x, y, 0xFFFFFF);
			return (NULL);
		}
		coor.x /= 3;
		coor.y /= 3;
		data->i++;
	}
	data->i = data->max_it;
	put_pixel(data->surf, x, y, 0xAA0000);
	return (NULL);
}

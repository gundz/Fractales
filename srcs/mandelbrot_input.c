/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mandelbrot_input.c                                 :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/03/16 19:21:22 by fgundlac          #+#    #+#             */
/*   Updated: 2015/03/16 19:21:23 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <fractol.h>

// int						mandelbrot_move(unsigned int button, unsigned int key,
// 	t_data *data)
// {
// 	int					ret;

// 	ret = 0;
// 	if (key == K_LEFT)
// 		ret += !!(data->fract.pos.x -= data->fract.pos_inc);
// 	if (key == K_RIGHT)
// 		ret += !!(data->fract.pos.x += data->fract.pos_inc);
// 	if (key == K_UP)
// 		ret += !!(data->fract.pos.y -= data->fract.pos_inc);
// 	if (key == K_DOWN)
// 		ret += !!(data->fract.pos.y += data->fract.pos_inc);
// 	return (ret);
// 	(void)button;
// }

int						mandelbrot_zoom(unsigned int button, unsigned int key,
	t_data *data)
{
	int					ret;

	ret = 0;
	if (button == M_LEFT || key == K_KPLUS)
	{
		data->fract.zoom *= ZOOM;
		data->fract.zoomp.x *= ZOOM;
		data->fract.zoomp.y *= ZOOM;
		data->fract.zoomp.x += ((RX - RXZ) / 2) / ZOOM - \
			(data->mlx.m_x - RX2) * ZOOM + (data->mlx.m_x - RX2);
		data->fract.zoomp.y += ((RY - RYZ) / 2) / ZOOM - \
			(data->mlx.m_y - RY2) * ZOOM + (data->mlx.m_y - RY2);
		data->fract.max_it *= 1.01;
		ret++;
	}
	if (button == M_RIGHT || key == K_KMIN)
	{
		data->fract.zoom /= ZOOM;
		data->fract.zoomp.x /= ZOOM;
		data->fract.zoomp.y /= ZOOM;
		data->fract.zoomp.x -= ((RX - RXZ) / 2) / ZOOM - \
			(data->mlx.m_x - RX2) * ZOOM + (data->mlx.m_x - RX2);
		data->fract.zoomp.y -= ((RY - RYZ) / 2) / ZOOM - \
			(data->mlx.m_y - RY2) * ZOOM + (data->mlx.m_y - RY2);
		data->fract.max_it /= 1.01;
		ret++;
	}
	return (ret);
}

int						mandelbrot_input(unsigned int button, unsigned int key,
	t_data *data)
{
	int					ret;

	ret = 0;
	//ret += mandelbrot_move(button, key, data);
	ret += mandelbrot_zoom(button, key, data);
	return (ret);
}

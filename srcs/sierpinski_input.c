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

static int				zoom_in(t_data *data)
{
	data->fract->data.zoom *= ZOOM;
	data->fract->data.zoomp.x *= ZOOM;
	data->fract->data.zoomp.y *= ZOOM;
	data->fract->data.zoomp.x += ((RX - RXZ) / 2) / ZOOM - \
		(data->mlx.m_x - RX2) * ZOOM + (data->mlx.m_x - RX2);
	data->fract->data.zoomp.y += ((RY - RYZ) / 2) / ZOOM - \
		(data->mlx.m_y - RY2) * ZOOM + (data->mlx.m_y - RY2);
	data->fract->data.max_it *= 1.01;
	return (1);
}

static int				zoom_out(t_data *data)
{
	data->fract->data.zoom /= ZOOM;
	data->fract->data.zoomp.x /= ZOOM;
	data->fract->data.zoomp.y /= ZOOM;
	data->fract->data.zoomp.x -= ((RX - RXZ) / 2) / ZOOM - \
		(data->mlx.m_x - RX2) * ZOOM + (data->mlx.m_x - RX2);
	data->fract->data.zoomp.y -= ((RY - RYZ) / 2) / ZOOM - \
		(data->mlx.m_y - RY2) * ZOOM + (data->mlx.m_y - RY2);
	data->fract->data.max_it /= 1.01;
	return (1);
}

int						sierpinski_input(unsigned int button, unsigned int key,
	t_data *data)
{
	int					ret;

	ret = 0;
	if (button == M_LEFT || key == M_S_UP)
		ret += zoom_in(data);
	if (button == M_RIGHT || key == M_S_DOWN)
		ret += zoom_out(data);
	if (key == K_U)
	{
		data->fract->data.max_it = 42;
		ret++;
	}
	if (key == K_I)
	{
		data->fract->data.max_it *= 2;
		ret++;
	}
	return (ret);
}

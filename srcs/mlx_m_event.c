/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mlx_m_event.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/01 04:21:03 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 06:55:27 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <mlx_lib.h>
#include <fractol.h>

int				mlx_m_event(unsigned int button, int x, int y, void *param)
{
	int			ret;
	t_data		*data;

	ret = 0;
	data = param;
	data->mlx.m_x = x;
	data->mlx.m_y = y;
	ret += mandelbrot_m_input(button, &data->fract);
	if (ret > 0)
		main_mlx(data);
	return (0);
}

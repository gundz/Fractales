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

int				mlx_m_button(unsigned int button, int x, int y, void *param)
{
	int			ret;
	t_data		*data;

	ret = 0;
	data = param;
	data->mlx.m_x = x;
	data->mlx.m_y = y;
	ret += mandelbrot_input(button, -1, data);
	if (ret > 0)
		main_mlx(data);
	return (0);
}

int				mlx_m_move(int x, int y, void *param)
{
	t_data		*data;

	data = param;
	data->mlx.m_x = x;
	data->mlx.m_y = y;
	return (0);
}

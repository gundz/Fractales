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

int						mandelbrot_move(unsigned int key, t_fract *data)
{
	int					ret;

	ret = 0;
	if (key == K_LEFT)
		ret += !!(data->pos.x -= data->pos_inc);
	if (key == K_RIGHT)
		ret += !!(data->pos.x += data->pos_inc);
	if (key == K_UP)
		ret += !!(data->pos.y -= data->pos_inc);
	if (key == K_DOWN)
		ret += !!(data->pos.y += data->pos_inc);
	return (ret);
}

int						mandelbrot_zoom(unsigned int button, t_fract *data)
{
	int					ret;

	ret = 0;
	if (button == M_S_UP)
	{
		data->zoom.x += 1 + data->zoom_inc;
		data->zoom.y += 1 + data->zoom_inc;
		data->zoom_inc *= 1.05;
		data->pos_inc *= 0.95;
		data->nb_it += 1;
		data->nb_it *= 1.01;
		ret += 1;
	}
	if (button == M_S_DOWN)
	{
		data->zoom.x -= 1 + data->zoom_inc;
		data->zoom.y -= 1 + data->zoom_inc;
		data->zoom_inc /= 1.05;
		data->pos_inc /= 0.95;
		data->nb_it -= 1;
		data->nb_it -= 1 * 0.99;
		ret += 1;
	}
	return (ret);
}

int						mandelbrot_k_input(unsigned int key, t_fract *data)
{
	int					ret;

	ret = 0;
	ret += mandelbrot_move(key, data);
	return (ret);
}

int						mandelbrot_m_input(unsigned int button, t_fract *data)
{
	int					ret;

	ret = 0;
	ret += mandelbrot_zoom(button, data);
	return (ret);
}
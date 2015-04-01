/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fractol.h                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/01 04:58:18 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 04:58:18 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef FRACTOL_H
# define FRACTOL_H

# include <mlx_lib.h>

typedef struct		s_v2d
{
	long double		x;
	long double		y;
}					t_v2d;

typedef struct		s_fract
{
	int				i;
	int				nb_it;
	t_v2d			pos;
	t_v2d			m_pos;
	long double		pos_inc;
	t_v2d			zoom;
	long double		zoom_inc;
	t_v2d			c;
	t_mlx_surf		*surf;
}					t_fract;

void				*mandelbrot(void *arg, const int x, const int y);
void				init_mandelbrot(t_fract *data);

#endif

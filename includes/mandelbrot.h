/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mandelbrot.h                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/17 16:00:34 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/17 16:00:35 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef MANDELBROT_H
# define MANDELBROT_H

typedef struct	s_mandelbrot
{
	double		zoom;
	double		cx;
	double		cy;
	double		maxiteration;
	double		movex;
	double		movey;
	double		oldcx;
	double		oldcy;
	int			mx;
	int			my;
}				t_mandelbrot;

#endif

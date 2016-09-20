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
	double		x1;
	double		x2;
	double		y1;
	double		y2;
	double		zoomx;
	double		zoomy;
	double		maxiteration;
}				t_mandelbrot;

#endif

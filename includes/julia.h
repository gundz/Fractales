/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   julia.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/17 16:00:32 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/17 16:00:33 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef JULIA_H
# define JULIA_H

typedef struct	s_julia
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
}				t_julia;

#endif

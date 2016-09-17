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
	int			mx;
	int			my;
	double		zoom;
	double		movex;
	double		movey;
	int			maxiteration;
}				t_julia;

#endif

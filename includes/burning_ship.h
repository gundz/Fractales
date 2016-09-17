/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   burning_ship.h                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/17 16:00:25 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/17 16:00:26 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef BURNING_SHIP_H
# define BURNING_SHIP_H

typedef struct	s_burning
{
	double		zoom;
	double		movex;
	double		movey;
	int			maxiteration;
}				t_burning;

#endif

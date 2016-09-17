/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   tricorn.h                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/17 16:00:36 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/17 16:00:37 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef TRICORN_H
# define TRICORN_H

typedef struct	s_tricorn
{
	double		zoom;
	double		movex;
	double		movey;
	int			maxiteration;
}				t_tricorn;

#endif

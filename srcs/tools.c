/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   tools.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/02 05:46:35 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/02 05:46:36 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <fractol.h>

t_v2d			set_v2d(long double x, long double y, t_v2d *v)
{
	t_v2d		ret;

	ret.x = x;
	ret.y = y;
	if (v != NULL)
	{
		v->x = x;
		v->y = y;
	}
	return (ret);
}

t_v2i			set_v2i(int x, int y, t_v2i *v)
{
	t_v2i		ret;

	ret.x = x;
	ret.y = y;
	if (v != NULL)
	{
		v->x = x;
		v->y = y;
	}
	return (ret);
}

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   tga_tools.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/03 13:16:06 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/03 13:16:07 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <mlx_lib.h>
#include <stdlib.h>

unsigned int					*surf_to_int(t_mlx_surf *surf)
{
	unsigned int				i;
	unsigned int				j;
	unsigned int				*ret;

	if (!(ret = (unsigned int *)malloc((sizeof(unsigned int) *
		surf->x) * surf->y)))
		return (NULL);
	i = 0;
	while (i < surf->y)
	{
		j = 0;
		while (j < surf->x)
		{
			ret[(surf->x * i) + j] = get_color(surf, j, i);
			j++;
		}
		i++;
	}
	return (ret);
}

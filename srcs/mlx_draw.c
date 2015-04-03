/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mlx_draw.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/02 08:50:58 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/02 08:50:59 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <mlx_lib.h>
#include <mlx.h>

void				put_pixel(t_mlx_surf *surf,
		const int x, const int y, const int color)
{
	int				res;

	res = (x * (surf->bpp / sizeof(char *)) + (y * surf->pitch));
	if ((x <= RX && y <= RY && x > 0 && y > 0) && res >= 0)
	{
		surf->pixels[res] = color;
		surf->pixels[res + 1] = (color >> 8);
		surf->pixels[res + 2] = (color >> 16);
	}
}

unsigned int		get_color(t_mlx_surf *surf, const int x, const int y)
{
	int				res;
	unsigned int	ret;
	unsigned int	r;
	unsigned int	g;
	unsigned int	b;

	ret = 0;
	res = (x * (surf->bpp / sizeof(char *)) + (y * surf->pitch));
	r = (surf->pixels[res] >> 16) & 0xFF;
	g = (surf->pixels[res] >> 8) & 0xFF;
	b = (surf->pixels[res]) & 0xFF;
	ret = (ret << 8) + r;
	ret = (ret << 8) + g;
	ret = (ret << 8) + b;
	return (ret);
}

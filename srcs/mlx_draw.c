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
		surf->pixels[res] = color & 0xFF;
		surf->pixels[res + 1] = (color >> 8) & 0xFF;
		surf->pixels[res + 2] = (color >> 16) & 0xFF;
	}
}

unsigned int		get_color_from_surf(t_mlx_surf *surf,
	const int x, const int y)
{
	int				res;
	unsigned int	ret;

	ret = 0;
	res = (x * (surf->bpp / sizeof(char *)) + (y * surf->pitch));
	ret |= (surf->pixels[res]) & 0x0000FF;
	ret |= (surf->pixels[res + 1] << 8) & 0x00FF00;
	ret |= (surf->pixels[res + 2] << 16) & 0xFF0000;
	return (ret);
}

unsigned int		rgb_to_uint(const int r, const int g, const int b)
{
	unsigned int	ret;

	ret = 0;
	ret |= (b) & 0x0000FF;
	ret |= (g << 8) & 0x00FF00;
	ret |= (r << 16) & 0xFF0000;
	return (ret);
}

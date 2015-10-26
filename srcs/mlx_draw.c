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

void				draw_line(t_mlx_surf *surf, t_v2i a, t_v2i b,
	const int color)
{
	t_v2i			d;
	t_v2i			s;
	int				err;
	int				e2;

	d.x = ABS(b.x - a.x);
	d.y = ABS(b.y - a.y);
	s.x = (a.x < b.x ? 1 : -1);
	s.y = (a.y < b.y ? 1 : -1);
	err = d.x - d.y;
	while (a.x != b.x || a.y != b.y)
	{
		put_pixel(surf, a.x, a.y, color);
		if ((e2 = 2 * err) > -d.y)
			err -= (d.y + 0 * (a.x += s.x));
		if (a.x == b.x && a.y == b.y)
		{
			put_pixel(surf, a.x, a.y, color);
			return ;
		}
		if (e2 < d.x)
			err += (d.x + 0 * (a.y += s.y));
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

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mlx_surface.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/01 01:08:42 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 01:38:18 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <mlx_lib.h>
#include <stdlib.h>
#include <mlx.h>

void			put_pixel(t_mlx_surf *surf,
		const int x, const int y, const int color)
{
	int			res;

	res = (x * (surf->bpp / 8) + (y * surf->pitch));
	if ((x <= RX && y <= RY && x > 0 && y > 0) && res >= 0)
	{
		surf->pixels[res] = color;
		surf->pixels[res + 1] = (color >> 8);
		surf->pixels[res + 2] = (color >> 16);
	}
}

t_mlx_surf		*mlx_create_rgb_surface(void *const mlx,
	const int x, const int y, const int color)
{
	t_mlx_surf	*surf;
	int			i;
	int			j;

	if (!(surf = (t_mlx_surf *)malloc(sizeof(t_mlx_surf))))
		return (NULL);
	surf->img = mlx_new_image(mlx, x, y);
	surf->pixels =
		mlx_get_data_addr(surf->img, &surf->bpp, &surf->pitch, &surf->endian);
	i = 0;
	while (i < y)
	{
		j = 0;
		while (j < x)
		{
			put_pixel(surf, j, i, color);
			j++;
		}
		i++;
	}
	return (surf);
}

void			mlx_free_surface(void *mlx, t_mlx_surf *surf)
{
	mlx_destroy_image(mlx, surf->img);
	free(surf);
}

void			mlx_show_surf(t_mlx *mlx, t_mlx_surf *surf)
{
	mlx_put_image_to_window(mlx->mlx, mlx->win, surf->img, 0, 0);
}

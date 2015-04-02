/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mlx_init.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/01 01:35:01 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 01:35:02 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <mlx_lib.h>
#include <mlx.h>
#include <stdlib.h>

unsigned int	mlx_l_init(t_mlx *mlx, char *title, const int x, const int y)
{
	if (!(mlx->mlx = mlx_init()))
		return (-1);
	if (x != -1 && y != -1 && x >= 0 && y >= 0)
	{
		if (!(mlx->win = mlx_new_window(mlx->mlx, x, y, title)))
			return (-1);
	}
	else
	{
		if (!(mlx->win = mlx_new_window(mlx->mlx, RX, RY, title)))
			return (-1);
	}
	mlx->surf = mlx_create_rgb_surface(mlx->mlx, RX, RY, 0x000000);
	return (0);
}

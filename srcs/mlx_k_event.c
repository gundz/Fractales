/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mlx_event.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/01 02:52:37 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 02:52:38 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <mlx_lib.h>
#include <stdlib.h>
#include <mlx.h>

static void		quit(t_mlx *mlx)
{
	mlx_free_surface(mlx->mlx, mlx->surf);
	mlx_destroy_window(mlx->mlx, mlx->win);
	exit(0);
}

int				mlx_k_event(unsigned int key, t_data *param)
{
	if (key == K_ESC)
		quit(&param->mlx);
	return (0);
}

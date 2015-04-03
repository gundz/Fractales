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

#include <libft.h>
#include <mlx_lib.h>
#include <fractol.h>
#include <stdlib.h>
#include <mlx.h>
#include <time.h>

static void			quit(t_mlx *mlx)
{
	mlx_free_surface(mlx->mlx, mlx->surf);
	mlx_destroy_window(mlx->mlx, mlx->win);
	exit(0);
}

static void			take_screen(t_data *data)
{
	unsigned int	*screen;
	char			*name;
	char			*tmp;

	screen = surf_to_int(data->mlx.surf);
	tmp = ft_itoa(time(NULL));
	name = ft_strijoin(3, "screen/screen-", tmp, ".tga");
	write_tga(name, screen, data->mlx.surf->x, data->mlx.surf->y);
	free(screen);
	free(name);
	free(tmp);
}

int					mlx_k_press(unsigned int key, void *param)
{
	int				ret;
	t_data			*data;

	ret = 0;
	data = param;
	if (key == K_ESC)
		quit(&data->mlx);
	if (key == K_P)
		take_screen(data);
	ret += mandelbrot_input(-1, key, data);
	if (ret > 0)
		main_mlx(data);
	return (0);
}

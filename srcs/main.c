/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/01 01:08:38 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 01:08:39 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <libft.h>
#include <thread.h>
#include <mlx_lib.h>
#include <fractol.h>
#include <mlx.h>

void				main_mlx(t_data *data)
{
	t_thread		*thread;
	t_tab			*tab = get_t_tab(RX, RY, 0);
	t_fract			fract;

	init_mandelbrot(&fract);
	fract.surf = data->mlx.surf;
	thread = get_thread(5, tab, &fract, &mandelbrot);
	thread_it(thread);
	mlx_show_surf(&data->mlx, data->mlx.surf);
}

int					main(void)
{
	t_data			data;

	if (mlx_l_init(&data.mlx, "Fractol", -1, -1) != 0)
		ft_putstr_fd("Error while initializing MLX", 2);
	main_mlx(&data);
	mlx_key_hook(data.mlx.win, &mlx_k_event, &data.mlx);
	mlx_mouse_hook(data.mlx.win, &mlx_m_event, &data.mlx);
	mlx_loop(data.mlx.mlx);
	return (0);
}

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
	thread_it(data->thread);
	mlx_show_surf(&data->mlx, data->mlx.surf);
	mlx_string_put(data->mlx.mlx, data->mlx.win, 0, 0, 0xFF0000, ft_itoa(data->fract.max_it));
}

void				init_data(t_data *data)
{
	init_mandelbrot(&data->fract);
	data->fract.surf = data->mlx.surf;
	data->tab = get_t_tab(RX, RY, 0);
	data->thread = get_thread(NB_THREAD, data->tab, &data->fract, &mandelbrot);
}

int					main(void)
{
	t_data			data;

	if (mlx_l_init(&data.mlx, "Fractol", -1, -1) != 0)
		ft_putstr_fd("Error while initializing MLX", 2);
	init_data(&data);
	main_mlx(&data);
	mlx_hook(data.mlx.win, MotionNotify, PointerMotionMask, &mlx_m_move, &data);
	mlx_key_hook(data.mlx.win, &mlx_k_press, &data);
	mlx_mouse_hook(data.mlx.win, &mlx_m_button, &data);
	mlx_loop(data.mlx.mlx);
	return (0);
}

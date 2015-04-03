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

int					main_mlx(t_data *data)
{
	int				i;

	i = 0;
	data->fract = &data->fracts[i];
	data->thread->f = data->fract->fract;
	data->thread->data = &data->fract->data;
	thread_it(data->thread);
	mlx_show_surf(&data->mlx, data->fract->data.surf);
	return (0);
}

void				init_data(t_data *data)
{
	data->tab = get_t_tab(RX, RY, 0);
	data->thread = get_thread(NB_THREAD, data->tab, NULL, NULL);
	init_mandelbrot(&data->fracts[0].data);
	data->fracts[0].data.surf =
		mlx_create_rgb_surface(data->mlx.mlx, RX, RY, 0x000000);
	data->fracts[0].fract = &mandelbrot;
	data->fracts[0].input = &mandelbrot_input;
}

int					main(void)
{
	t_data			data;

	if (mlx_l_init(&data.mlx, "Fractol", -1, -1) != 0)
		ft_putstr_fd("Error while initializing MLX", 2);
	init_data(&data);
	main_mlx(&data);
	mlx_expose_hook(data.mlx.win, &main_mlx, &data);
	mlx_hook(data.mlx.win, MOTION, MOTION_MASK, &mlx_m_move, &data);
	mlx_key_hook(data.mlx.win, &mlx_k_press, &data);
	mlx_mouse_hook(data.mlx.win, &mlx_m_button, &data);
	mlx_loop(data.mlx.mlx);
	return (0);
}

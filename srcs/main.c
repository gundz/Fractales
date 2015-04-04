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
	data->thread->f = data->fract->fract;
	data->thread->data = &data->fract->data;
	if (data->fract->thread_status == 1)
		thread_it(data->thread);
	else
		data->thread->f(&data->fract->data, -1, -1);
	mlx_show_surf(&data->mlx, data->fract->data.surf);
	return (0);
}

void				init_data(t_data *data, const int init)
{
	data->fract_i = init - 1;
	change_fract(data);
	data->tab = get_t_tab(RX, RY, 0);
	data->thread = get_thread(NB_THREAD, data->tab, NULL, NULL);
	init_mandelbrot(&data->fracts[0].data);
	data->fracts[0].data.surf =
		mlx_create_rgb_surface(data->mlx.mlx, RX, RY, 0x000000);
	data->fracts[0].fract = &mandelbrot;
	data->fracts[0].input = &mandelbrot_input;
	data->fracts[0].thread_status = 1;
	init_julia(&data->fracts[1].data);
	data->fracts[1].data.surf =
		mlx_create_rgb_surface(data->mlx.mlx, RX, RY, 0x000000);
	data->fracts[1].fract = &julia;
	data->fracts[1].input = &julia_input;
	data->fracts[1].thread_status = 1;
	init_sierpinski(&data->fracts[2].data);
	data->fracts[2].data.surf =
		mlx_create_rgb_surface(data->mlx.mlx, RX, RY, 0x000000);
	data->fracts[2].fract = &sierpinski;
	data->fracts[2].input = &sierpinski_input;
	data->fracts[2].thread_status = 1;
}

int					get_option(int argc, char **argv)
{
	if (argc != 2)
		return (-1);
	if (ft_strcmp(argv[1], "mandelbrot") == 0)
		return (0);
	if (ft_strcmp(argv[1], "julia") == 0)
		return (1);
	if (ft_strcmp(argv[1], "sierpinski") == 0)
		return (2);
	return (-1);
}

int					main(int argc, char **argv)
{
	t_data			data;
	int				option;

	if ((option = get_option(argc, argv)) == -1)
	{
		ft_putstr_fd("Usage : ./fractol [mandelbrot][julia][sierpinski]\n", 2);
		return (-1);
	}
	if (mlx_l_init(&data.mlx, "Fractol", -1, -1) != 0)
		ft_putstr_fd("Error while initializing MLX", 2);
	init_data(&data, option);
	main_mlx(&data);
	mlx_expose_hook(data.mlx.win, &main_mlx, &data);
	mlx_hook(data.mlx.win, MOTION, MOTION_MASK, &mlx_m_move, &data);
	mlx_key_hook(data.mlx.win, &mlx_k_press, &data);
	mlx_mouse_hook(data.mlx.win, &mlx_m_button, &data);
	mlx_loop(data.mlx.mlx);
	return (0);
}

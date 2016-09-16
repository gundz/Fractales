/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main_check.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:36:22 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:36:23 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <easy_sdl.h>
#include <libft.h>
#include <header.h>

int					check_arg(int argc, char **argv, t_data *data)
{
	if (argc != 2)
		return (-1);
	if (ft_strstr(argv[1], "-mandelbrot"))
		data->fractal = MANDELBROT;
	else if (ft_strstr(argv[1], "-julia"))
		data->fractal = JULIA;
	else if (ft_strstr(argv[1], "-burning"))
		data->fractal = BURNING_SHIP;
	else if (ft_strstr(argv[1], "-tricorn"))
		data->fractal = TRICORN;
	else
		return (-1);
	return (0);
}

int					show_usage(void)
{
	ft_putstr("usage: ./fractol -[mandelbrot][julia][burning][tricorn]\n");
	return (0);
}

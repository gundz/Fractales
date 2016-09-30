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
	if (ft_strcmp(argv[1], "-mandelbrot") == 0)
		data->fractal_choose = MANDELBROT;
	else if (ft_strcmp(argv[1], "-julia") == 0)
		data->fractal_choose = JULIA;
	else if (ft_strcmp(argv[1], "-burning") == 0)
		data->fractal_choose = BURNING_SHIP;
	else if (ft_strcmp(argv[1], "-mandelbrot3") == 0)
		data->fractal_choose = MANDELBROT3;
	else if (ft_strcmp(argv[1], "-mandelbrot4") == 0)
		data->fractal_choose = MANDELBROT4;
	else if (ft_strcmp(argv[1], "-tricorn") == 0)
		data->fractal_choose = TRICORN;
	else
		return (-1);
	return (0);
}

int					show_usage(void)
{
	ft_putstr("usage: ./fractol ");
	ft_putstr("-[mandelbrot][julia][burning]");
	ft_putstr("[mandelbrot3][mandelbrot4][tricorn]");
	ft_putstr("\n");
	return (0);
}

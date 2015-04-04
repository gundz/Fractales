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

static void			quit(t_data *data)
{
	int				i;

	i = 0;
	free_thread(data->thread);
	while (i < NB_FRACT)
		mlx_free_surface(data->mlx.mlx, data->fracts[i++].data.surf);
	mlx_destroy_window(data->mlx.mlx, data->mlx.win);
	exit(0);
}

static void			take_screen(t_data *data)
{
	unsigned int	*screen;
	char			*name;
	char			*tmp;

	if (!(screen = surf_to_int(data->fract->data.surf)))
		return ;
	tmp = ft_itoa(time(NULL));
	name = ft_strijoin(3, "screen/screen-", tmp, ".tga");
	if (write_tga(name, screen,
		data->fract->data.surf->x, data->fract->data.surf->y) == -1)
		ft_putstr_fd("Failed to take screenshot\n", 2);
	else
	{
		ft_putstr("Screen saved under : ");
		ft_putstr(name);
		ft_putstr("\n");
	}
	free(screen);
	free(name);
	free(tmp);
}

int					change_fract(t_data *data)
{
	data->fract_i += 1;
	if (data->fract_i >= NB_FRACT)
		data->fract_i = 0;
	data->fract = &data->fracts[data->fract_i];
	return (1);
}

int					mlx_k_press(unsigned int key, void *param)
{
	int				ret;
	t_data			*data;

	ret = 0;
	data = param;
	if (key == K_ESC)
		quit(data);
	if (key == K_V)
		ret += change_fract(data);
	if (key == K_P)
		take_screen(data);
	ret += data->fract->input(-1, key, data);
	if (ret > 0)
		main_mlx(data);
	return (0);
}

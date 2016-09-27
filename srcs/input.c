/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   input.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/22 19:30:38 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/22 19:30:39 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <header.h>
#include <libft.h>
#include <time.h>

static int			check_input_list(t_data *data, \
	const int *key, const int *button)
{
	int				i;
	int				ret;

	ret = 0;
	i = 1;
	while (i <= key[0])
	{
		if (data->esdl->en.in.key[key[i]] == 1)
			ret++;
		i++;
	}
	i = 1;
	while (i <= button[0])
	{
		if (data->esdl->en.in.button[button[i]] == 1)
			ret++;
		i++;
	}
	return (ret);
}

void				change_fractal(t_data *data)
{
	if (data->esdl->en.in.key[SDL_SCANCODE_Q] == 1)
	{
		data->fractal_choose--;
		data->esdl->en.in.key[SDL_SCANCODE_Q] = 0;
	}
	else if (data->esdl->en.in.key[SDL_SCANCODE_W] == 1)
	{
		data->fractal_choose++;
		data->esdl->en.in.key[SDL_SCANCODE_W] = 0;
	}
	if (data->fractal_choose >= SIZE)
		data->fractal_choose = 0;
	else if (data->fractal_choose < 0)
		data->fractal_choose = SIZE;
}

void				take_screenshot(t_data *data)
{
	char			*timestamp;
	char			*name;

	timestamp = ft_itoa(time(NULL));
	name = ft_strjoin(timestamp, ".tga");
	free(timestamp);
	ft_putstr("Saving Screenshot, please wait...\n");
	write_tga(name, data->surf->pixels, SDL_RX, SDL_RY);
	ft_putstr("Screenshot saved under: \"");
	ft_putstr(name);
	ft_putstr("\"\n");
	free(name);
}

int					check_input(t_data *data)
{
	const int		key[] = {10, SDL_SCANCODE_LEFT, SDL_SCANCODE_RIGHT, \
		SDL_SCANCODE_UP, SDL_SCANCODE_DOWN, \
		SDL_SCANCODE_KP_PLUS, SDL_SCANCODE_KP_MINUS, \
		SDL_SCANCODE_A, SDL_SCANCODE_S, \
		SDL_SCANCODE_Q, SDL_SCANCODE_W};
	const int		button[] = {2, SDL_BUTTON_LEFT, SDL_BUTTON_RIGHT};
	int				ret;

	ret = 0;
	ret += check_input_list(data, key, button);
	if (data->esdl->en.in.key[SDL_SCANCODE_P] == 1)
		take_screenshot(data);
	change_fractal(data);
	return (ret);
}

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fractals_tools.c                                   :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/03 16:30:43 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/03 16:30:44 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <fractol.h>

void				change_fract(t_data *data)
{
	data->fract_i += 1;
	if (data->fract_i >= NB_FRACT)
		data->fract_i = 0;
	data->fract = &data->fracts[data->fract_i];
}

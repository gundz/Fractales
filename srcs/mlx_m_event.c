/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mlx_m_event.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/01 04:21:03 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 04:21:04 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <mlx_lib.h>

int				mlx_m_event(unsigned int button, int x, int y, t_data *param)
{
	param->mlx.m_x = x;
	param->mlx.m_y = y;
	return (0);
}

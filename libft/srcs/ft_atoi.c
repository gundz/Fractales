/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_atoi.c                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <fgundlac@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/10/08 20:48:23 by fgundlac          #+#    #+#             */
/*   Updated: 2015/10/08 20:48:24 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <stddef.h>

int			ft_atoi(const char *str)
{
	int		i;
	int		sign;
	int		result;

	if (str == NULL)
		return (0);
	sign = 1;
	i = 0;
	result = 0;
	while (str[i] == ' ' || str[i] == '\f' || str[i] == '\n'
			|| str[i] == '\r' || str[i] == '\t' || str[i] == '\v')
		i++;
	if (str[i] == '-' || str[i] == '+')
	{
		if (str[i] == '-')
			sign = -1;
		i++;
	}
	while (str[i] >= '0' && str[i] <= '9')
	{
		result = 10 * result + (str[i] - '0');
		i++;
	}
	return (result * sign);
}

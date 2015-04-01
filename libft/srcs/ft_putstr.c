/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_putstr.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2014/11/11 18:29:27 by fgundlac          #+#    #+#             */
/*   Updated: 2014/11/11 18:29:28 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <unistd.h>

void					ft_putstr(const char *const str)
{
	unsigned int		len;

	len = 0;
	while (str[len] != '\0')
		len++;
	write(1, str, len);
}

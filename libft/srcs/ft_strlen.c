/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strlen.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2014/11/11 18:31:07 by fgundlac          #+#    #+#             */
/*   Updated: 2014/11/11 18:31:08 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <stddef.h>

size_t							ft_strlen(const char *const str)
{
	unsigned int				len;

	len = 0;
	while (str[len] != '\0')
		++len;
	return (len);
}

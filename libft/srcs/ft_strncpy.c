/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strncpy.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <fgundlac@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2013/11/19 19:29:42 by fgundlac          #+#    #+#             */
/*   Updated: 2013/12/01 21:40:22 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <stddef.h>

char					*ft_strncpy(char *dest, char *src, size_t n)
{
	int					i;

	i = 0;
	while (src[i] != '\0' && n > 0)
	{
		dest[i] = src[i];
		i++;
		n--;
	}
	dest[i] = '\0';
	while (n > 0)
	{
		dest[i++] = '\0';
		n--;
	}
	return (dest);
}

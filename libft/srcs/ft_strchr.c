/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strchr.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <fgundlac@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/10/13 12:17:24 by fgundlac          #+#    #+#             */
/*   Updated: 2015/10/13 12:17:24 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <stddef.h>

char				*ft_strchr(const char *src, int c)
{
	while (*src != c)
	{
		if (*src == '\0')
			return (NULL);
		src++;
	}
	return ((char *)src);
}

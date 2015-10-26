/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strstr.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <fgundlac@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/10/08 20:46:57 by fgundlac          #+#    #+#             */
/*   Updated: 2015/10/08 20:46:58 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <stddef.h>

char			*ft_strstr(const char *str, const char *to_find)
{
	int			i;
	int			j;
	int			c;

	i = 0;
	c = 0;
	if (to_find[0] == '\0')
		return (char*)(str);
	while (str[i] != '\0')
	{
		j = i;
		c = 0;
		while (str[j] == to_find[c])
		{
			j++;
			c++;
			if (to_find[c] == '\0')
				return (char*)(&str[i]);
		}
		i++;
	}
	return (NULL);
}

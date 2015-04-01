/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strdup.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2014/11/11 18:30:11 by fgundlac          #+#    #+#             */
/*   Updated: 2014/11/11 18:30:12 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <stdlib.h>

char					*ft_strdup(const char *const str)
{
	unsigned int		len;
	char				*ret;

	len = 0;
	while (str[len] != '\0')
		len++;
	if (!(ret = (char *)malloc(sizeof(char) * (len + 1))))
		return (NULL);
	len = 0;
	while (str[len] != '\0')
	{
		ret[len] = str[len];
		len++;
	}
	ret[len] = '\0';
	return (ret);
}

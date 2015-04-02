/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strrev.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2013/12/20 14:57:24 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 23:37:03 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <libft.h>
#include <stdlib.h>

char			*ft_strrev(char *str)
{
	char		*rev;
	size_t		len;
	int			i;

	i = 0;
	len = ft_strlen(str);
	if (!(rev = (char *)malloc(sizeof(char) * (len + 1))))
		return (NULL);
	while (len--)
	{
		rev[i] = str[len];
		i++;
	}
	rev[i] = '\0';
	return (rev);
}

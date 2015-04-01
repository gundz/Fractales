/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strijoin.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/03/18 09:53:35 by fgundlac          #+#    #+#             */
/*   Updated: 2015/03/18 09:55:29 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <libft.h>
#include <stdarg.h>
#include <stdlib.h>

static unsigned int		join_len(va_list ap, unsigned int n)
{
	va_list				ap2;
	unsigned int		len;

	va_copy(ap2, ap);
	len = 0;
	while (n--)
		len += ft_strlen(va_arg(ap2, char *));
	return (len);
}

char					*ft_strijoin(unsigned int n, ...)
{
	va_list				ap;
	unsigned int		len;
	char				*ret;
	char				*tmp;
	unsigned int		i;

	va_start(ap, n);
	len = join_len(ap, n);
	if (!(ret = (char *)malloc(sizeof(char) * (len + 1))))
		return (NULL);
	i = 0;
	while (n--)
	{
		tmp = va_arg(ap, char *);
		while (*tmp != '\0')
			ret[i++] = *tmp++;
	}
	ret[i] = '\0';
	va_end(ap);
	return (ret);
}

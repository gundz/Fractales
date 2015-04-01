/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_putendl_fd.c                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2014/11/11 18:29:07 by fgundlac          #+#    #+#             */
/*   Updated: 2014/11/11 18:29:08 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <unistd.h>

void					ft_putendl_fd(const char *const str, int fd)
{
	unsigned int		len;

	len = 0;
	while (str[len] != '\0')
		len++;
	write(fd, str, len);
	write(fd, "\n", 1);
}

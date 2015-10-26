/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_ctablen.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/03/18 09:53:16 by fgundlac          #+#    #+#             */
/*   Updated: 2015/03/18 09:53:18 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <stddef.h>

unsigned int				ft_ctablen(char **const tab)
{
	unsigned int			len;

	len = 0;
	while (tab[len] != NULL)
		len++;
	return (len);
}

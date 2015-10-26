/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strlen.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/03/17 21:00:46 by fgundlac          #+#    #+#             */
/*   Updated: 2015/03/17 21:01:20 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

int					ft_strlen(const char *const str)
{
	int				len;

	len = 0;
	while (str[len] != '\0')
		len++;
	return (len);
}

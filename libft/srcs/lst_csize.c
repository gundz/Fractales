/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lst_csize.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/03/17 21:01:49 by fgundlac          #+#    #+#             */
/*   Updated: 2015/03/17 21:01:50 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <libft.h>
#include <stddef.h>

int						lst_csize(t_list *lst)
{
	t_list				*lstwalker;
	char				*tmp;
	int					size;
	int					i;

	size = 0;
	lstwalker = lst;
	while (lstwalker != NULL)
	{
		tmp = lstwalker->data;
		i = 0;
		while (tmp[i] != '\0')
			i++;
		size += i;
		if (lstwalker->next == NULL)
			break ;
		lstwalker = lstwalker->next;
	}
	return (size);
}

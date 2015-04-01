/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lst_count.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2014/11/11 18:31:18 by fgundlac          #+#    #+#             */
/*   Updated: 2014/11/11 18:31:19 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <libft.h>
#include <stddef.h>

unsigned int					lst_count(t_list *lst)
{
	t_list						*lstwalker;
	unsigned int				ret;

	ret = 0;
	lstwalker = lst;
	while (lstwalker != NULL)
	{
		++ret;
		if (lstwalker->next == NULL)
			break ;
		lstwalker = lstwalker->next;
	}
	return (ret);
}

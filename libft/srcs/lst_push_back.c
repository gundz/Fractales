/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lst_push_back.c                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2014/11/11 18:31:24 by fgundlac          #+#    #+#             */
/*   Updated: 2014/11/11 18:31:25 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <libft.h>
#include <stddef.h>

void					lst_push_back(t_list **lst, void *data)
{
	t_list				*lstwalker;

	if (data == NULL)
		return ;
	if (*lst == NULL)
	{
		*lst = lst_create_elem(data);
		return ;
	}
	lstwalker = *lst;
	while (lstwalker != NULL)
	{
		if (lstwalker->next == NULL)
			break ;
		lstwalker = lstwalker->next;
	}
	lstwalker->next = lst_create_elem(data);
}

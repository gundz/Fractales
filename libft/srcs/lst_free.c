/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lst_free.c                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/03/18 09:38:04 by fgundlac          #+#    #+#             */
/*   Updated: 2015/03/18 09:38:05 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <libft.h>
#include <stdlib.h>

void				lst_free(t_list *lst, const int free_data)
{
	t_list			*tmp;

	while (lst != NULL)
	{
		if (lst->next == NULL)
			break ;
		tmp = lst->next;
		if (free_data == 1)
			free(lst->data);
		free(lst);
		lst = tmp;
	}
	free(lst);
}

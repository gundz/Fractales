/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   t_tab_thread.c                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/03 16:07:02 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/03 16:09:11 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <thread.h>
#include <stdlib.h>

t_tab				*get_t_tab(const int x, const int y, const int init)
{
	t_tab			*tab;
	int				i;
	int				j;

	if (!(tab = (t_tab *)malloc(sizeof(t_tab))))
		return (NULL);
	if (!(tab->tab = (int **)malloc(sizeof(int *) * y)))
		return (NULL);
	i = 0;
	while (i < y)
	{
		if (!(tab->tab[i] = (int *)malloc(sizeof(int) * x)))
			return (NULL);
		j = 0;
		while (j < x)
			tab->tab[i][j++] = init;
		i++;
	}
	tab->size_x = x;
	tab->size_y = y;
	return (tab);
}

void				free_t_tab(t_tab *tab)
{
	int				i;

	i = 0;
	while (i < tab->size_y)
	{
		free(tab->tab[i]);
		i++;
	}
	free(tab->tab);
	free(tab);
}

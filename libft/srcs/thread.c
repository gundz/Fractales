/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   thread.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/03/16 17:04:58 by fgundlac          #+#    #+#             */
/*   Updated: 2015/03/16 17:30:18 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <thread.h>
#include <stdlib.h>

void				*loop(void *arg)
{
	t_thread		*v;

	v = arg;
	v->y = v->y_init;
	while (v->y < v->y_max)
	{
		v->x = 0;
		while (v->x < v->tab->size_x)
		{
			v->f(arg, v->x, v->y);
			v->x++;
		}
		v->y++;
	}
	free(arg);
	return (NULL);
}

t_thread			*get_thread(const int nb_thread, t_tab *const tab,
		void *const data, void *(*f)(void *, int, int))
{
	t_thread		*ret;

	if (!(ret = (t_thread *)malloc(sizeof(t_thread))))
		return (NULL);
	ret->nb_thread = nb_thread;
	ret->f = f;
	ret->tab = tab;
	ret->y_init = 0;
	ret->y_max = 0;
	ret->data = data;
	return (ret);
}

void				thread_it(t_thread *thread)
{
	pthread_t		threads[thread->nb_thread];
	int				i;
	t_thread		*tmp;

	i = 0;
	while (i < thread->nb_thread)
	{
		tmp = get_thread(thread->nb_thread, thread->tab, thread->data,
				thread->f);
		tmp->y_init = (thread->tab->size_y / thread->nb_thread) * i;
		tmp->y_max = (thread->tab->size_y / thread->nb_thread) * (i + 1);
		if (pthread_create(&threads[i], NULL, loop, tmp) == -1)
			return ;
		i++;
	}
	i = 0;
	while (i < thread->nb_thread)
	{
		if (pthread_join(threads[i], NULL) == -1)
			return ;
		i++;
	}
}

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
